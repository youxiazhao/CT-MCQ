#!/usr/bin/env python3
"""
CT-MCQ Exam Pipeline — 3 models × 2 reasoning efforts = 6 configurations
=========================================================================
For each case the pipeline sends:
  1. Clinical text  — everything BEFORE "discussion:" in the case text (no answer leakage)
  2. CT images      — uniformly sampled PNGs, byte-limited, CONSISTENT across all 9 runs
  3. Options A–E    — identified_final_diagnosis + rank 1–4 DDx candidates, shuffled once

Configurations (model × reasoning effort):
  gpt-5.4       × none / medium
  gpt-5.4-mini  × none / medium
  gpt-5.4-nano  × none / medium

  "none"   → Responses API call without a reasoning parameter
  "medium" → Responses API with reasoning={"effort": "medium"}

Input  (required):
  --input PATH       sampled_*.json produced by sample_from_ddx.py
                     image_folders inside it must be absolute paths to PNG slice directories
  --output-dir PATH  directory where all output files are written

Output:
  <output-dir>/questions.json         — MCQ question set (built once, reused by all configs)
  <output-dir>/<config_key>.json      — per-config results, checkpointed after every batch
  <output-dir>/all_exam_results.json  — accuracy summary across all configs

Usage:
    python exam_pipeline.py \\
        --input     results/exam_800/sampled_800.json \\
        --output-dir results/exam_800

    python exam_pipeline.py \\
        --input results/exam_800/sampled_800.json --output-dir results/exam_800 \\
        --dry-run                               # validate setup, skip API calls

    python exam_pipeline.py \\
        --questions-file results/exam_800/questions_800.json \\
        --output-dir results/exam_800           # skip question building

    python exam_pipeline.py \\
        --input results/exam_800/sampled_800.json --output-dir results/exam_800 \\
        --configs gpt54_low gpt54mini_none      # run only selected configs

See README.md for full input/output schema and environment setup.
"""

import asyncio
import argparse
import base64
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent          # CT-MCQ/scripts/
REPO_ROOT    = SCRIPT_DIR.parent                        # CT-MCQ/
ENV_FILE     = REPO_ROOT / ".env"                       # CT-MCQ/.env  (all keys: OpenAI + Langfuse)
DEFAULT_QUESTIONS_FILE = REPO_ROOT / "input" / "question_example.json"  # ready-to-run example (questions format)
DEFAULT_OUTPUT         = REPO_ROOT / "examples" / "output"               # example output
LOGS_DIR     = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE     = LOGS_DIR / "exam_pipeline.log"

# ── Parameters ───────────────────────────────────────────────────────────────
TARGET_SLICES      = 85
MAX_TOTAL_IMAGES   = 400
BYTE_LIMIT_ENCODED = 38 * 1024 * 1024   # 38 MB encoded
MAX_OUTPUT_TOKENS  = 4096            # structured reasoning JSON needs room
MAX_OUTPUT_TOKENS_REASONING = 16384  # reasoning-effort models need more headroom
RANDOM_SEED        = 42
DEFAULT_CONCURRENCY = 16
MAX_RETRIES        = 4     # API call retries (exponential backoff on rate-limit)
RETRY_BASE_DELAY   = 2.0   # seconds; delay doubles each attempt

CONFIGS = [
    {"model": "gpt-5.4",      "effort": "none",   "key": "gpt54_none"},
    {"model": "gpt-5.4",      "effort": "medium", "key": "gpt54_medium"},
    {"model": "gpt-5.4-mini", "effort": "none",   "key": "gpt54mini_none"},
    {"model": "gpt-5.4-mini", "effort": "medium", "key": "gpt54mini_medium"},
    {"model": "gpt-5.4-nano", "effort": "none",   "key": "gpt54nano_none"},
    {"model": "gpt-5.4-nano", "effort": "medium", "key": "gpt54nano_medium"},
]

LETTERS = list("ABCDE")

# EXAM_DIR is set at runtime (defaults to DEFAULT_OUTPUT; overridden by --output-dir)
EXAM_DIR = DEFAULT_OUTPUT

# ── Logging ──────────────────────────────────────────────────────────────────
EXAM_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── System prompt + structured output schema (Responses API json_schema) ───
@contextmanager
def _noop_ctx():
    yield


# Strict JSON Schema for OpenAI Structured Outputs (additionalProperties: false;
# every key in properties must be listed in required).
RESPONSE_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_option": {
            "type": "string",
            "enum": ["A", "B", "C", "D", "E"],
            "description": "The chosen option letter.",
        },
        "selected_diagnosis": {
            "type": "string",
            "description": "Exact text of the chosen option from the list.",
        },
        "confidence": {
            "type": "integer",
            "description": "Integer 0–100 reflecting residual uncertainty after reasoning.",
        },
        "difficulty": {
            "type": "string",
            "enum": ["easy", "medium", "hard"],
            "description": (
                "Estimated difficulty of this MCQ: "
                "'easy' = answer clear from imaging/history alone; "
                "'medium' = requires careful multi-step analysis; "
                "'hard' = genuinely ambiguous, rare entity, or subtle imaging findings."
            ),
        },
        "reasoning": {
            "type": "object",
            "additionalProperties": False,
            "description": "Structured radiological reasoning in fixed sections.",
            "properties": {
                "per_group_findings": {
                    "type": "array",
                    "description": "One entry per imaging group/series (same order as listed in the prompt).",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "group_index": {
                                "type": "integer",
                                "description": "1-based index matching the imaging group list order.",
                            },
                            "group_label": {
                                "type": "string",
                                "description": "Folder/series name or window (e.g. portal venous axial, lung window).",
                            },
                            "findings": {
                                "type": "string",
                                "description": "Synthesized findings across all slices in this group (not slice-by-slice).",
                            },
                        },
                        "required": ["group_index", "group_label", "findings"],
                    },
                },
                "integrated_imaging_impression": {
                    "type": "string",
                    "description": "Synthesized imaging findings across slices (location, morphology, relevant negatives).",
                },
                "clinical_correlation": {
                    "type": "string",
                    "description": "How imaging ties to presentation, demographics, timeline, risk factors.",
                },
                "option_analysis": {
                    "type": "object",
                    "additionalProperties": False,
                    "description": "Rule-in / rule-out for each letter vs imaging + history.",
                    "properties": {
                        "A": {"type": "string"},
                        "B": {"type": "string"},
                        "C": {"type": "string"},
                        "D": {"type": "string"},
                        "E": {"type": "string"},
                    },
                    "required": ["A", "B", "C", "D", "E"],
                },
                "final_rationale": {
                    "type": "string",
                    "description": "Why the selected option is best and why the others are less likely.",
                },
            },
            "required": [
                "per_group_findings",
                "integrated_imaging_impression",
                "clinical_correlation",
                "option_analysis",
                "final_rationale",
            ],
        },
    },
    "required": ["selected_option", "selected_diagnosis", "confidence", "difficulty", "reasoning"],
}


SYSTEM_PROMPT = """\
You are an expert radiologist taking a diagnostic imaging board examination.

You will be given:
• A clinical case description (presentation and patient background only — \
imaging findings and discussion are NOT included)
• CT scan images (uniformly sampled slices from multiple anatomical planes / \
window settings)
• A multiple-choice list of 5 possible diagnoses labelled A through E

Your task: select the SINGLE BEST diagnosis from the provided options based \
solely on the clinical information and CT images.

Fill every field in the required JSON schema:
• reasoning.per_group_findings — exactly one object per imaging group listed in the \
prompt (group_index 1…N in that order); group_label should match that row; \
synthesize findings across all slices in that group, not per-slice.
• reasoning.integrated_imaging_impression — consolidated imaging picture (location, \
laterality, morphology, important negatives).
• reasoning.clinical_correlation — link to history/exam; what narrows the differential.
• reasoning.option_analysis — for each letter A–E, concise support vs counter-evidence.
• reasoning.final_rationale — tie-break: why your chosen letter wins over the rest.
• confidence — align with true residual uncertainty (e.g., overlap between two options).
• difficulty — judge this question's difficulty AFTER you finish reasoning:
  - "easy": the correct diagnosis is evident from imaging + history with minimal ambiguity.
  - "medium": correct answer requires careful integration of imaging features and clinical details.
  - "hard": case is genuinely ambiguous, involves a rare entity, or requires advanced subspecialty knowledge.

selected_diagnosis must match the exact wording of your chosen option. Output must \
conform to the JSON schema — no prose outside it, no markdown fences.
"""


# ── Env loader ───────────────────────────────────────────────────────────────

def load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


# ── Image utilities ──────────────────────────────────────────────────────────

def get_sorted_images(folder: str) -> list[Path]:
    p = Path(folder)
    if not p.exists():
        return []
    try:
        return sorted(p.glob("*.png"), key=lambda x: int(x.stem))
    except ValueError:
        return sorted(p.glob("*.png"))


def uniform_sample(files: list, n: int) -> list:
    total = len(files)
    if total == 0:
        return []
    if total <= n:
        return files
    if n == 1:
        return [files[0]]
    indices = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})
    return [files[i] for i in indices]


def get_file_size(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            return f.tell()
    except Exception:
        return 400 * 1024


def byte_limited_sample(files: list, encoded_byte_limit: int) -> list:
    if not files:
        return []
    if len(files) > MAX_TOTAL_IMAGES:
        files = uniform_sample(files, MAX_TOTAL_IMAGES)
    raw_sizes = [get_file_size(f) for f in files]
    total_raw = sum(raw_sizes)
    raw_limit = int(encoded_byte_limit * 3 / 4)
    if total_raw <= raw_limit:
        return files
    lo, hi = 1, len(files)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        indices = sorted({round(i * (len(files) - 1) / (mid - 1)) for i in range(mid)}) if mid > 1 else [0]
        if sum(raw_sizes[i] for i in indices) <= raw_limit:
            lo = mid
        else:
            hi = mid - 1
    indices = sorted({round(i * (len(files) - 1) / (lo - 1)) for i in range(lo)}) if lo > 1 else [0]
    return [files[i] for i in indices]


def encode_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Text helper ──────────────────────────────────────────────────────────────

def extract_clinical_text(text: str) -> str:
    """Return only Title + presentation + patient — before discussion."""
    for marker in ["\n\n\ndiscussion:", "\ndiscussion:", "\n\n\n\nstudy_findings:"]:
        idx = text.lower().find(marker.lower())
        if idx != -1:
            return text[:idx].strip()
    return text.strip()


# ── Question builder ─────────────────────────────────────────────────────────

def build_questions(data: dict, seed: int = RANDOM_SEED) -> dict:
    """
    Build MCQ questions from sampled_50.json.
    Returns { case_key: question_dict }.
    """
    rng = random.Random(seed)
    cases = data["cases"]
    questions = {}

    for case_key, case in cases.items():
        clinical_text = extract_clinical_text(case.get("text", ""))

        ddx = case.get("differential_diagnosis", {})
        correct_answer = ddx.get("identified_final_diagnosis", "")
        candidates = ddx.get("candidates", [])

        all_options = [correct_answer]
        for cand in candidates[:4]:
            name = cand.get("diagnosis_name", "").strip()
            if name and name not in all_options:
                all_options.append(name)

        idx = 4
        while len(all_options) < 5 and idx < len(candidates):
            name = candidates[idx].get("diagnosis_name", "").strip()
            if name and name not in all_options:
                all_options.append(name)
            idx += 1

        rng.shuffle(all_options)

        options = {LETTERS[i]: dx for i, dx in enumerate(all_options)}
        correct_letter = next(
            (letter for letter, dx in options.items() if dx == correct_answer),
            None,
        )

        questions[case_key] = {
            "clinical_text": clinical_text,
            "options": options,
            "correct_answer": correct_answer,
            "correct_letter": correct_letter,
            "image_folders": case.get("image_folders", []),
        }

    return questions


# ── API input builder ────────────────────────────────────────────────────────

def format_imaging_groups_block(question: dict) -> str:
    """
    List imaging groups (one per image_folders entry) so the model fills
    reasoning.per_group_findings with one row per group, not per slice.
    """
    folders = question.get("image_folders") or []
    stats = (question.get("image_stats") or {}).get("folders") or {}
    if not folders:
        return (
            "## Imaging groups\n\n"
            "1. All provided CT images (single group).\n\n"
            "Provide exactly one per_group_findings entry (group_index 1).\n\n"
        )
    lines: list[str] = []
    for i, folder in enumerate(folders, start=1):
        name = Path(folder).name
        s = stats.get(name)
        if s:
            detail = f"{s['sampled']} slices sampled from {s['total']} in this folder"
        else:
            detail = "multiple slices sampled"
        lines.append(f"{i}. {name} — {detail}")
    body = "\n".join(lines)
    return (
        "## Imaging groups\n\n"
        "The examination includes the following groups (series / window / reconstruction). "
        "Slices below usually appear in folder order (group 1, then 2, …); if globally "
        "subsampled for size, still write one synthesized summary per group listed.\n\n"
        f"{body}\n\n"
        "Your JSON must include exactly one per_group_findings object per row above, "
        "same order, group_index 1 through N.\n\n"
    )


def _group_images_by_folder(
    image_paths: list[str], image_folders: list[str],
) -> list[tuple[str, list[int]]]:
    """
    Map each image path back to its source folder and return
    (folder_name, [indices_into_image_paths]) in folder order.
    Images whose folder is unknown are appended as a final "other" group.
    """
    folder_to_indices: dict[str, list[int]] = {}
    for idx, p in enumerate(image_paths):
        parent = Path(p).parent.name
        folder_to_indices.setdefault(parent, []).append(idx)

    ordered: list[tuple[str, list[int]]] = []
    seen = set()
    for folder in image_folders:
        name = Path(folder).name
        if name in folder_to_indices:
            ordered.append((name, folder_to_indices[name]))
            seen.add(name)
    for name, indices in folder_to_indices.items():
        if name not in seen:
            ordered.append((name, indices))
    return ordered


def build_responses_input(question: dict, encoded_images: list[str]) -> list[dict]:
    """Build Responses API input payload with per-group image separators."""
    clinical_text = question["clinical_text"]
    options = question["options"]
    image_paths = question.get("image_paths") or []
    image_folders = question.get("image_folders") or []

    options_str = "\n".join(f"{letter}. {dx}" for letter, dx in options.items())

    groups_block = format_imaging_groups_block(question)
    intro_text = (
        f"## Clinical Presentation\n\n{clinical_text}\n\n"
        f"{groups_block}"
        f"## CT Images ({len(encoded_images)} slices total)\n\n"
    )

    content: list[dict] = [{"type": "input_text", "text": intro_text}]

    grouped = _group_images_by_folder(image_paths, image_folders)
    if grouped and len(grouped) > 1:
        for group_idx, (folder_name, indices) in enumerate(grouped, start=1):
            content.append({
                "type": "input_text",
                "text": f"--- Group {group_idx}: {folder_name} ({len(indices)} slices) ---",
            })
            for i in indices:
                if i < len(encoded_images):
                    content.append({
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encoded_images[i]}",
                        "detail": "low",
                    })
    else:
        for b64 in encoded_images:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
                "detail": "low",
            })

    question_text = (
        f"## Question\n\n"
        f"Based on the clinical information and CT images above, "
        f"select the SINGLE BEST diagnosis:\n\n"
        f"{options_str}\n\n"
        "Return ONLY the JSON object as specified — no markdown, no extra text."
    )
    content.append({"type": "input_text", "text": question_text})

    return [{"role": "user", "content": content}]


# ── Single case processor ───────────────────────────────────────────────────

async def process_case(
    client,
    trace,                          # per-config Langfuse Trace, or None
    semaphore: asyncio.Semaphore,
    case_key: str,
    question: dict,
    encoded_images: list[str],
    model: str,
    effort: str,
    config_key: str,
    dry_run: bool,
) -> dict:
    result = {
        "case_key": case_key,
        "config": config_key,
        "model": model,
        "effort": effort,
        "options": question["options"],
        "correct_letter": question["correct_letter"],
        "correct_answer": question["correct_answer"],
        "selected_option": None,
        "selected_diagnosis": None,
        "is_correct": None,
        "confidence": None,
        "difficulty": None,
        "reasoning": None,
        "metadata": {},
    }

    if dry_run:
        result["metadata"]["dry_run"] = True
        return result

    # One span per case, nested inside the per-config trace's root span.
    span = trace.start_observation(
        name=case_key,
        as_type="span",
        metadata={"model": model, "effort": effort,
                  "num_images": len(encoded_images)},
    ) if trace else None

    async with semaphore:
        try:
            with _noop_ctx():
                input_payload = build_responses_input(question, encoded_images)

                out_tokens = MAX_OUTPUT_TOKENS_REASONING if effort != "none" else MAX_OUTPUT_TOKENS
                api_kwargs = {
                    "model": model,
                    "instructions": SYSTEM_PROMPT,
                    "input": input_payload,
                    "max_output_tokens": out_tokens,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "radiology_exam_mc_answer",
                            "strict": True,
                            "schema": RESPONSE_JSON_SCHEMA,
                            "description": "Board MCQ answer with per-group imaging and per-option reasoning.",
                        }
                    },
                }
                if effort != "none":
                    api_kwargs["reasoning"] = {"effort": effort}

                t0 = time.monotonic()
                for attempt in range(MAX_RETRIES):
                    try:
                        response = await client.responses.create(**api_kwargs)
                        break
                    except Exception as exc:
                        # Retry on rate-limit (429); surface everything else immediately.
                        is_rate_limit = (
                            "rate limit" in str(exc).lower()
                            or "429" in str(exc)
                            or type(exc).__name__ == "RateLimitError"
                        )
                        if not is_rate_limit or attempt == MAX_RETRIES - 1:
                            raise
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        log.warning(
                            f"  [{config_key}][{case_key}] Rate limit hit — "
                            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        await asyncio.sleep(delay)
                elapsed = time.monotonic() - t0

                raw = getattr(response, "output_text", "") or ""
                usage = response.usage

                # Strip markdown fences if present
                fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
                if fence:
                    raw = fence.group(1).strip()

                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                    result["metadata"]["parse_error"] = raw[:1000]

                if parsed:
                    sel = parsed.get("selected_option", "").strip().upper()
                    result["selected_option"] = sel
                    result["selected_diagnosis"] = parsed.get("selected_diagnosis", "")
                    result["is_correct"] = (sel == question["correct_letter"])
                    result["confidence"] = parsed.get("confidence")
                    result["difficulty"] = parsed.get("difficulty")
                    result["reasoning"] = parsed.get("reasoning")

                result["metadata"].update({
                    "api_time_s": round(elapsed, 2),
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens,
                })

        except Exception as e:
            log.warning(f"  [{config_key}][{case_key}] Error: {e}")
            result["metadata"]["error"] = str(e)
            if span:
                span.update(level="ERROR", status_message=str(e))
                span.end()
            return result

    if span:
        span.update(output={"is_correct": result.get("is_correct"),
                            "difficulty": result.get("difficulty"),
                            "confidence": result.get("confidence")})
        span.end()
    return result


# ── Config runner ────────────────────────────────────────────────────────────

def resolve_image_paths(image_folders: list[str]) -> tuple[list[str], dict]:
    """
    Sample images from folders and apply byte limit.
    Returns (list_of_path_strings, image_stats).
    Deterministic: same folders → same images every time.
    """
    all_sampled = []
    folder_stats = {}
    for folder in image_folders:
        folder_name = Path(folder).name
        all_files = get_sorted_images(folder)
        sampled = uniform_sample(all_files, TARGET_SLICES)
        folder_stats[folder_name] = {"total": len(all_files), "sampled": len(sampled)}
        all_sampled.extend(sampled)

    final = byte_limited_sample(all_sampled, BYTE_LIMIT_ENCODED)
    stats = {
        "folders": folder_stats,
        "total_sampled": len(all_sampled),
        "final_count": len(final),
        "byte_capped": len(final) < len(all_sampled),
    }
    return [str(p) for p in final], stats


def encode_image_paths(paths: list[str]) -> list[str]:
    """Encode a list of image file paths to base64 strings."""
    return [encode_b64(Path(p)) for p in paths if Path(p).exists()]


def load_all_config_results(configs: list[dict]) -> dict[str, dict]:
    """Load existing results for all configs. Returns {config_key: {case_key: result}}."""
    all_existing = {}
    for cfg in configs:
        key = cfg["key"]
        path = EXAM_DIR / f"{key}.json"
        if path.exists():
            with open(path) as f:
                all_existing[key] = json.load(f)
        else:
            all_existing[key] = {}
    return all_existing


def save_config_result(config_key: str, data: dict) -> None:
    path = EXAM_DIR / f"{config_key}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def run_all_parallel(
    client, langfuse, configs: list[dict], questions: dict,
    concurrency: int, dry_run: bool,
) -> dict[str, dict]:
    """
    Run all configs in parallel per case.
    For each batch of cases: encode images once, then fire API calls
    for ALL configs simultaneously.
    """
    all_existing = load_all_config_results(configs)

    # Resolve image paths once, cache in questions
    questions_file = EXAM_DIR / "questions.json"
    paths_resolved = False
    if not dry_run:
        for ck, q in questions.items():
            if "image_paths" not in q:
                paths, stats = resolve_image_paths(q["image_folders"])
                q["image_paths"] = paths
                q["image_stats"] = stats
                paths_resolved = True
        if paths_resolved:
            with open(questions_file, "w") as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)
            log.info("Image paths resolved and cached in questions.json")

    # Build todo list: (case_key, config) pairs that still need processing.
    # Re-run if: missing, dry-run, has error, or is_correct is None (parse failure).
    case_keys = list(questions.keys())
    todo_per_case: dict[str, list[dict]] = {}
    for ck in case_keys:
        pending = []
        for cfg in configs:
            existing_r = all_existing[cfg["key"]].get(ck, {})
            needs_run = (
                not existing_r
                or existing_r.get("metadata", {}).get("dry_run")
                or existing_r.get("metadata", {}).get("error")
                or (existing_r.get("is_correct") is None
                    and not existing_r.get("metadata", {}).get("dry_run"))
                or existing_r.get("difficulty") is None
            )
            if needs_run:
                pending.append(cfg)
        if pending:
            todo_per_case[ck] = pending

    total_calls = sum(len(v) for v in todo_per_case.values())
    log.info(f"Total pending: {len(todo_per_case)} cases × configs = {total_calls} API calls")

    if not todo_per_case:
        log.info("All cases already completed for all configs.")
        return all_existing

    semaphore = asyncio.Semaphore(concurrency)
    case_batch_size = max(1, concurrency // max(len(configs), 1))
    todo_case_keys = list(todo_per_case.keys())

    # One Langfuse trace per config: cases appear as observations inside it.
    # In Langfuse v4 a top-level observation IS the trace; the root span's
    # name becomes the trace name shown in the UI.
    config_traces: dict[str, object] = {}
    if langfuse and not dry_run:
        for cfg in configs:
            config_traces[cfg["key"]] = langfuse.start_observation(
                name=f"exam-{cfg['key']}",
                as_type="span",
                metadata={
                    "model": cfg["model"],
                    "effort": cfg["effort"],
                    "total_cases": len(questions),
                    "pending_cases": len(todo_per_case),
                },
            )

    for batch_start in range(0, len(todo_case_keys), case_batch_size):
        batch_keys = todo_case_keys[batch_start:batch_start + case_batch_size]

        # Encode images once per case (shared across all configs).
        # Run in threads so base64 I/O doesn't block the event loop.
        encoded_cache = {}
        if not dry_run:
            pairs = await asyncio.gather(*[
                asyncio.to_thread(
                    encode_image_paths, questions[ck].get("image_paths", [])
                )
                for ck in batch_keys
            ])
            encoded_cache = dict(zip(batch_keys, pairs))

        # Fire all (case × config) pairs in this batch
        tasks = []
        task_meta = []
        for ck in batch_keys:
            encoded = encoded_cache.get(ck, [])
            for cfg in todo_per_case[ck]:
                tasks.append(
                    process_case(
                        client, config_traces.get(cfg["key"]), semaphore,
                        ck, questions[ck],
                        encoded, cfg["model"], cfg["effort"], cfg["key"], dry_run,
                    )
                )
                task_meta.append((ck, cfg["key"]))

        results = await asyncio.gather(*tasks)

        # Distribute results to per-config stores
        for r, (ck, cfg_key) in zip(results, task_meta):
            r["metadata"]["image_stats"] = questions[ck].get("image_stats", {})
            all_existing[cfg_key][ck] = r

        # Checkpoint all config files (parallel async writes to avoid blocking).
        await asyncio.gather(*[
            asyncio.to_thread(save_config_result, cfg["key"], all_existing[cfg["key"]])
            for cfg in configs
        ])

        done_total = sum(len(v) for v in all_existing.values())
        expected_total = len(questions) * len(configs)
        log.info(f"  Batch [{batch_start+1}..{batch_start+len(batch_keys)}] done | "
                 f"overall: {done_total}/{expected_total} case×config pairs")

        del encoded_cache

    # Close all per-config root spans so traces appear as "completed" in Langfuse.
    for root_span in config_traces.values():
        try:
            root_span.end()
        except Exception:
            pass

    return all_existing


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MCQ Exam Pipeline — 9 configurations")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--configs", nargs="+", default=None,
                   help="Run only these config keys (e.g. gpt54_low gpt54mini_none)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--input", type=str, default=None,
                   help="Path to sampled_*.json (default: results/sampled_50.json). "
                        "Ignored when --questions-file is given.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Custom output directory (default: results/exam)")
    p.add_argument("--questions-file", type=str, default=None,
                   help="Path to a pre-built questions JSON file (skips building from sampled data)")
    return p.parse_args()


async def async_main():
    global EXAM_DIR

    args = parse_args()

    if args.output_dir:
        EXAM_DIR = Path(args.output_dir)
        if not EXAM_DIR.is_absolute():
            EXAM_DIR = SCRIPT_DIR / args.output_dir
        EXAM_DIR.mkdir(parents=True, exist_ok=True)

    load_env(ENV_FILE)

    # Resolve --input / --questions-file, falling back to the bundled example.
    input_file = None
    if args.input:
        input_file = Path(args.input)
        if not input_file.is_absolute():
            input_file = SCRIPT_DIR / args.input

    questions_file_arg = args.questions_file
    if not questions_file_arg and not input_file:
        # Default: use the bundled example (questions format) so the pipeline
        # runs out of the box without any extra arguments.
        questions_file_arg = str(DEFAULT_QUESTIONS_FILE)

    data = {}
    if input_file and not questions_file_arg:
        with open(input_file) as f:
            data = json.load(f)

    # ── Build questions (once, shared across all configs) ─────────────────
    questions_file = EXAM_DIR / "questions.json"
    if questions_file_arg:
        custom_qfile = Path(questions_file_arg)
        if not custom_qfile.is_absolute():
            custom_qfile = SCRIPT_DIR / questions_file_arg
        with open(custom_qfile) as f:
            questions = json.load(f)
        log.info(f"Loaded {len(questions)} questions from: {custom_qfile}")
    elif questions_file.exists():
        with open(questions_file) as f:
            questions = json.load(f)
        log.info(f"Loaded {len(questions)} pre-built questions")
    else:
        log.info(f"Building questions from {input_file.name} ...")
        questions = build_questions(data, seed=args.seed)
        with open(questions_file, "w") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        log.info(f"Built and saved {len(questions)} questions")

    valid_q = {k: v for k, v in questions.items() if v.get("correct_letter")}
    log.info(f"Valid questions (with correct answer): {len(valid_q)}")

    # ── Select configs to run ─────────────────────────────────────────────
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c["key"] in args.configs]
    else:
        configs_to_run = CONFIGS

    log.info(f"Configs to run: {[c['key'] for c in configs_to_run]}")
    log.info(f"Concurrency: {args.concurrency}  Dry-run: {args.dry_run}")

    # ── Create client ─────────────────────────────────────────────────────
    client = None
    langfuse = None
    if not args.dry_run:
        langfuse_ok = bool(
            os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")
        )
        import httpx
        client_timeout = httpx.Timeout(300.0, connect=30.0)
        if langfuse_ok:
            try:
                from langfuse.openai import AsyncOpenAI
                from langfuse import Langfuse as _Langfuse
                client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=client_timeout,
                    max_retries=3,
                )
                langfuse = _Langfuse()   # stateful client — supports .trace() / .span()
                log.info("Langfuse tracing enabled.")
            except ImportError:
                log.warning("langfuse not installed — tracing disabled.")
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    timeout=client_timeout,
                    max_retries=3,
                )
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=client_timeout,
                max_retries=3,
            )
            log.info("No Langfuse keys — tracing disabled.")

    # ── Run all configs in parallel per case ─────────────────────────────
    t0 = time.monotonic()
    all_existing = await run_all_parallel(
        client, langfuse, configs_to_run, valid_q,
        args.concurrency, args.dry_run,
    )
    elapsed = time.monotonic() - t0
    log.info(f"\nAll configs completed in {elapsed:.1f}s")

    # ── Build summary ─────────────────────────────────────────────────────
    all_config_results = {}
    for cfg in configs_to_run:
        key = cfg["key"]
        results = all_existing.get(key, {})
        correct = sum(1 for v in results.values() if v.get("is_correct") is True)
        total = sum(1 for v in results.values() if v.get("is_correct") is not None)
        acc = correct / total if total else 0
        all_config_results[key] = {
            "model": cfg["model"],
            "effort": cfg["effort"],
            "correct": correct,
            "total": total,
            "accuracy": round(acc, 4),
        }

    summary = {
        "configs": {c["key"]: {"model": c["model"], "effort": c["effort"]} for c in CONFIGS},
        "total_questions": len(valid_q),
        "results": all_config_results,
    }
    combined_path = EXAM_DIR / "all_exam_results.json"
    with open(combined_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"\nCombined summary: {combined_path}")

    # ── Summary table ─────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("ACCURACY SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'Config':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    log.info("-" * 55)
    for key, r in all_config_results.items():
        log.info(f"{key:<25} {r['correct']:>8} {r['total']:>6} {r['accuracy']:>9.1%}")

    if langfuse:
        try:
            langfuse.flush()
            log.info("\nLangfuse buffer flushed.")
        except Exception:
            pass

    if client:
        await client.close()

    log.info("\nDone.")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
