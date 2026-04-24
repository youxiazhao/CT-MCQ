# CT-MCQ Exam Pipeline

A fully asynchronous multi-model multiple-choice exam pipeline for CT diagnostic imaging cases.

Each case is presented to **6 model configurations** (3 models × 2 reasoning efforts) via the OpenAI Responses API, with CT scan slices encoded as base64 images alongside structured clinical text. Per-config results are checkpointed after every batch; a final accuracy summary is emitted on completion.

---

## Repository Structure

```
CT-MCQ/
├── .env.example                 ← copy to .env and fill in your API keys
├── .gitignore
├── README.md
├── requirements.txt
├── input/
│   └── question_example.json    ← 1-case ready-to-run example (questions format)
├── scripts/
│   └── exam_pipeline.py         ← main pipeline
└── examples/
    └── output/                  ← reference output produced by the example input
        ├── gpt54nano_none.json
        └── all_exam_results.json
```

> `logs/`, `results/`, and `output/` are created at runtime and excluded from git.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run the bundled 1-case example (smallest/cheapest config)
python scripts/exam_pipeline.py --configs gpt54nano_none
```

With no input arguments the pipeline uses `input/question_example.json` and writes to `examples/output/`. You should get the same answer/output structure as the committed reference files.

---

## Requirements

- Python **3.9+**
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Environment Setup

The pipeline reads `.env` from the **repository root** (`CT-MCQ/.env`).

```bash
cp .env.example .env
```

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

The pipeline accepts **two input formats**, controlled by which CLI flag you pass.

### Option A — Pre-built questions (`--questions-file`)

This is the format used by the bundled example. Image paths are already resolved to individual PNG files.

```jsonc
{
  "<case_key>": {
    "clinical_text": "Title:...\n\npresentation:...\n\npatient:Age:40 years\nGender:Female",
    "options": {
      "A": "Diagnosis A", "B": "Diagnosis B", "C": "Diagnosis C",
      "D": "Diagnosis D", "E": "Diagnosis E"
    },
    "correct_answer": "Diagnosis A",
    "correct_letter": "A",
    "image_folders": [
      "/abs/path/to/ct_case/ct_quizze_XX/CASE_ID/Axial_portal_venous",
      "/abs/path/to/ct_case/ct_quizze_XX/CASE_ID/Axial_bone_window"
    ],
    "image_paths": [
      "/abs/path/to/.../Axial_portal_venous/0.png",
      "/abs/path/to/.../Axial_portal_venous/2.png"
      // ... already-sampled, ready-to-encode list of PNG files
    ],
    "image_stats": {
      "folders": { "Axial_portal_venous": { "total": 12, "sampled": 12 } },
      "total_sampled": 25, "final_count": 25, "byte_capped": false
    }
  }
}
```

See [`input/question_example.json`](input/question_example.json) for a complete real example.

### Option B — Sampled cases (`--input`)

Raw output from a sampling step. The pipeline builds questions on the fly (shuffles options, samples up to 85 PNGs per folder, applies a 38 MB encoded-byte cap).

```jsonc
{
  "total_sampled": 800,
  "sampling_criteria": { ... },
  "cases": {
    "<case_key>": {
      "text": "Title:...\n\npresentation:...\n\ndiscussion:...",
      // The pipeline extracts everything BEFORE the first "discussion:" marker as
      // the clinical text. Everything after is the answer key and is never sent.

      "image_folders": [
        "/abs/path/to/ct_case/ct_quizze_XX/CASE_ID/Axial_portal_venous"
      ],
      // Each folder contains numbered PNG slice files (0.png, 1.png, …).
      // Paths must be absolute and accessible on the machine running the pipeline.

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

`image_folders` (and `image_paths` if pre-resolved) must point to **actual files on disk**. If the JSON was produced on another machine, update the absolute paths before running.

---

## Running

All commands are run from the **repository root** (`CT-MCQ/`).

```bash
# Bundled example — no arguments needed
python scripts/exam_pipeline.py --configs gpt54nano_none

# Pre-built questions file + custom output dir
python scripts/exam_pipeline.py \
    --questions-file /path/to/questions_800.json \
    --output-dir     results/exam_800

# Raw sampled JSON (the pipeline builds questions automatically)
python scripts/exam_pipeline.py \
    --input      /path/to/sampled_800.json \
    --output-dir results/exam_800

# Dry run — validates setup without making any API calls
python scripts/exam_pipeline.py \
    --input      /path/to/sampled_800.json \
    --output-dir results/exam_800 \
    --dry-run

# Run only specific configs
python scripts/exam_pipeline.py \
    --questions-file /path/to/questions_800.json \
    --output-dir     results/exam_800 \
    --configs gpt54_medium gpt54mini_none gpt54nano_medium
```

### CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--questions-file PATH` | `input/question_example.json` | Pre-built questions JSON. Loaded as-is. |
| `--input PATH` | — | Raw sampled JSON. Pipeline builds questions from it. Mutually exclusive with `--questions-file`. |
| `--output-dir PATH` | `examples/output/` | Where all output files are written. Created automatically. |
| `--configs KEY …` | all 6 | Run only the listed config keys. |
| `--concurrency N` | `16` | Max simultaneous API calls in flight. |
| `--seed N` | `42` | Random seed for option shuffling (only matters for `--input` mode). |
| `--dry-run` | off | Build questions and plan the run, skip all API calls. |

> If neither `--input` nor `--questions-file` is given, the bundled example at `input/question_example.json` is used.

**Valid config keys** (for `--configs`):

```
gpt54_none       gpt54_medium
gpt54mini_none   gpt54mini_medium
gpt54nano_none   gpt54nano_medium
```

---

## Output

All files are written to `--output-dir` (default `examples/output/`).

```
<output-dir>/
├── questions.json          # MCQ set the pipeline used (cached for resume)
├── gpt54_none.json         # One file per requested model configuration
├── gpt54_medium.json
├── gpt54mini_none.json
├── gpt54mini_medium.json
├── gpt54nano_none.json
├── gpt54nano_medium.json
└── all_exam_results.json   # Accuracy summary across all run configs
```

A reference output of running `--configs gpt54nano_none` against the bundled example is committed at [`examples/output/`](examples/output/).

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
    "gpt54_none":     { "model": "gpt-5.4",      "effort": "none",   "correct": 512, "total": 800, "accuracy": 0.64 },
    "gpt54_medium":   { "model": "gpt-5.4",      "effort": "medium", "correct": 612, "total": 800, "accuracy": 0.765 },
    "gpt54mini_none": { "model": "gpt-5.4-mini", "effort": "none",   "correct": 480, "total": 800, "accuracy": 0.60 }
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

If both `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` are set in `.env`, the pipeline enables tracing automatically:

- One **trace** per model configuration (e.g. `exam-gpt54_medium`)
- One **span** per case nested inside its config trace
- Each span carries model / effort / image-count metadata and ends with `is_correct` / `difficulty` / `confidence` as output

Leave the keys empty (or unset) to disable tracing — this has zero effect on pipeline behavior.

> Requires `langfuse>=4.0`. The pipeline uses the v4 OpenTelemetry-style `start_observation` API.
