# CT-MCQ Exam Pipeline

A fully asynchronous multi-model multiple-choice exam pipeline for CT diagnostic imaging cases.

Each case is presented to **6 model configurations** (3 models × 2 reasoning efforts) via the OpenAI Responses API, with CT scan slices encoded as base64 images alongside structured clinical text. Per-config results are checkpointed after every batch; a final accuracy summary is emitted on completion.

---

## Repository Structure

```
CT-MCQ/
├── .env.example         ← copy to .env and fill in your API keys
├── .gitignore
├── README.md
├── requirements.txt
└── scripts/
    └── exam_pipeline.py ← main pipeline
```

> **`results/`**, **`logs/`**, and **`data/`** are created at runtime and excluded from git.

---

## Requirements

- Python **3.9+**

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

The pipeline reads `.env` from the **repository root** (`CT-MCQ/.env`).

```ini
# .env

# Required
OPENAI_API_KEY=sk-...

# Optional — Langfuse LLM tracing (leave empty to disable)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Input

### Sampled cases file (`--input`, required)

Pass a `sampled_*.json` file via `--input`. **There is no usable default** — this must always be specified.

**Expected JSON schema:**

```jsonc
{
  "total_sampled": 800,
  "sampling_criteria": { ... },   // metadata, not read by the pipeline
  "cases": {
    "<case_key>": {

      "text": "Title:...\n\npresentation:...\n\ndiscussion:...",
      // The pipeline extracts everything BEFORE the first "discussion:" marker
      // as the clinical text sent to the model. Everything after is the answer
      // key and is never sent.

      "image_folders": [
        "/absolute/path/to/ct_case/ct_quizze_XX/CASE_ID/Axial_portal_venous",
        "/absolute/path/to/ct_case/ct_quizze_XX/CASE_ID/Axial_bone_window"
      ],
      // Each entry is a folder of sequentially numbered PNG slice files
      // (0.png, 1.png, …). Paths must be absolute and accessible on the
      // machine running the pipeline.
      //
      // The pipeline uniformly samples up to 85 slices per folder, then
      // applies a 38 MB encoded-byte cap across all folders for the case.
      // If image_paths are pre-resolved (e.g. via build_exam_questions.py),
      // they are reused as-is.

      "differential_diagnosis": {
        "identified_final_diagnosis": "Correct diagnosis name",
        "candidates": [
          { "rank": 1, "diagnosis_name": "Top DDx candidate" }
          // ranks 1–4 fill the four distractor options; at least 4 needed
        ]
      }
    }
  }
}
```

### Image files

The paths inside `image_folders` must point to actual directories on disk. If the `sampled_*.json` was produced on a different machine, update the absolute paths before running.

### Pre-built questions file (`--questions-file`, optional)

If you have already resolved image paths with `build_exam_questions.py`, pass the result directly to skip the image-resolution step:

```bash
python scripts/exam_pipeline.py \
    --questions-file /path/to/questions_800.json \
    --output-dir     /path/to/results/exam_800
```

---

## Running

All commands are run from the **repository root** (`CT-MCQ/`).

```bash
# Run all 9 configs
python scripts/exam_pipeline.py \
    --input      /path/to/sampled_800.json \
    --output-dir results/exam_800

# Dry run — validates setup without making API calls
python scripts/exam_pipeline.py \
    --input      /path/to/sampled_800.json \
    --output-dir results/exam_800 \
    --dry-run

# Run only specific model configs
python scripts/exam_pipeline.py \
    --input      /path/to/sampled_800.json \
    --output-dir results/exam_800 \
    --configs gpt54_low gpt54mini_none gpt54nano_medium

# Use a pre-built questions file (skips image path resolution)
python scripts/exam_pipeline.py \
    --questions-file /path/to/questions_800.json \
    --output-dir     results/exam_800
```

### CLI Reference

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input PATH` | Yes* | — | Path to `sampled_*.json`. Absolute or relative to repo root. |
| `--output-dir PATH` | Yes* | `results/exam/` | Directory where all output files are written. Created automatically. |
| `--questions-file PATH` | No | — | Pre-built questions JSON. Skips question building entirely. |
| `--configs KEY …` | No | all 9 | Run only the listed config keys. |
| `--concurrency N` | No | `16` | Max simultaneous API calls in flight. |
| `--seed N` | No | `42` | Random seed for option shuffling. Changing it reassigns which letter is correct. |
| `--dry-run` | No | off | Build questions and plan the run, skip all API calls. |

*Not required when `--questions-file` is given, but strongly recommended to always set `--output-dir`.

**Valid config keys** (for `--configs`):

```
gpt54_none       gpt54_medium
gpt54mini_none   gpt54mini_medium
gpt54nano_none   gpt54nano_medium
```

---

## Output

All files are written to `--output-dir`. The directory is created if it does not exist.

```
<output-dir>/
├── questions.json          # MCQ set built from --input (cached; reused on resume)
├── gpt54_none.json         # Results for each of the 6 model configurations
├── gpt54_medium.json
├── gpt54mini_none.json
├── gpt54mini_medium.json
├── gpt54nano_none.json
├── gpt54nano_medium.json
└── all_exam_results.json   # Accuracy summary across all configs
```

### `questions.json`

Built once and shared across all 9 configs. Includes resolved image paths.

```jsonc
{
  "<case_key>": {
    "clinical_text": "...",        // text before "discussion:" — what the model sees
    "options": {                   // A–E shuffled from correct answer + top-4 DDx
      "A": "Diagnosis name",
      "B": "...", "C": "...", "D": "...", "E": "..."
    },
    "correct_answer": "Diagnosis name",
    "correct_letter": "B",        // letter assigned to the correct answer after shuffle
    "image_folders": ["..."],     // original folder paths from sampled JSON
    "image_paths": ["..."],       // resolved individual PNG paths (after sampling + byte cap)
    "image_stats": {
      "folders": {
        "Axial_portal_venous": { "total": 42, "sampled": 42 }
      },
      "total_sampled": 85,
      "final_count": 79,
      "byte_capped": false
    }
  }
}
```

### `<config_key>.json`

One entry per case. Checkpointed after every batch — safe to interrupt and resume.

```jsonc
{
  "<case_key>": {
    "case_key": "...",
    "config": "gpt54_none",
    "model": "gpt-5.4",
    "effort": "none",
    "options": { "A": "...", "B": "...", "C": "...", "D": "...", "E": "..." },
    "correct_letter": "B",
    "correct_answer": "...",
    "selected_option": "B",
    "selected_diagnosis": "...",
    "is_correct": true,
    "confidence": 82,              // integer 0–100
    "difficulty": "medium",        // "easy" | "medium" | "hard"
    "reasoning": {
      "per_group_findings": [      // one entry per image_folders entry
        { "group_index": 1, "group_label": "Axial_portal_venous", "findings": "..." }
      ],
      "integrated_imaging_impression": "...",
      "clinical_correlation": "...",
      "option_analysis": { "A": "...", "B": "...", "C": "...", "D": "...", "E": "..." },
      "final_rationale": "..."
    },
    "metadata": {
      "api_time_s": 14.3,
      "input_tokens": 18240,
      "output_tokens": 1024,
      "total_tokens": 19264,
      "image_stats": { ... }
    }
  }
}
```

### `all_exam_results.json`

```jsonc
{
  "total_questions": 800,
  "results": {
    "gpt54_none":   { "model": "gpt-5.4", "effort": "none",   "correct": 512, "total": 800, "accuracy": 0.64 },
    "gpt54_low":    { "model": "gpt-5.4", "effort": "low",    "correct": 548, "total": 800, "accuracy": 0.685 },
    ...
  }
}
```

---

## Concurrency Model

The pipeline uses `asyncio` with `asyncio.Semaphore(--concurrency)` to cap simultaneous API calls.

For each batch of cases, images are encoded in parallel via `asyncio.to_thread` (non-blocking). All `(case × config)` pairs in the batch are then launched concurrently, with the semaphore ensuring at most `--concurrency` calls are in flight at any time.

Rate-limit errors (HTTP 429) are retried up to 4 times with exponential backoff (2 s → 4 s → 8 s → 16 s). All other errors are logged and marked in `metadata.error`.

---

## Resuming an Interrupted Run

Results are checkpointed per config after every batch. On the next run with the same `--output-dir`, already-completed cases are detected and skipped — only pending or errored cases are re-processed.

---

## Logs

Logs are written to `CT-MCQ/logs/exam_pipeline.log` and streamed to stdout simultaneously.

---

## Langfuse Tracing (Optional)

If `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` are set in `.env`, the pipeline enables span tracing automatically:

- One **Trace** per model configuration (e.g. `exam-gpt54_low`)
- One **Span** per case nested inside its config trace

Leave both keys empty (or unset) to disable tracing entirely — this has no effect on pipeline behavior.
