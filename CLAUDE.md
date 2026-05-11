# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase studying when SFT (and GRPO) on a teacher's outputs preserves vs. degrades a student's general capabilities. Training is driven by YAML configs that point at JSON files of pre-generated teacher responses; runs SFT-train a student LoRA, evaluate on a fixed eval suite (MMLU, IFEval, MATH-500, Olympiads, optionally APPS), and emit a `summary.png` plotting losses + per-checkpoint scores.

All training and sampling go through Anthropic's [tinker](https://pypi.org/project/tinker/) SDK (`tinker.ServiceClient` / `TrainingClient` / `SamplingClient`). No local GPUs are used.

## Environment

- Python venv is symlinked: `.venv → /workspace/.venv` (use `.venv/bin/python` if PATH isn't set).
- `.env` (loaded by every entrypoint via `python-dotenv`) holds `TINKER_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`.
- No build/lint/test commands — this is a research repo with no test suite.

## Entry points

All scripts in [scripts/](scripts/) call `load_dotenv` and `sys.path.insert(0, REPO_ROOT)`, so always invoke them from the repo root:

| Command | Purpose |
| --- | --- |
| `python scripts/train.py <config.yaml>` | SFT a student on a stored teacher JSON, eval, plot. |
| `python scripts/rl_train.py <rl_config.yaml>` | GRPO on Olympiads or APPS, eval, plot. |
| `python scripts/run_with_self_distillation.py <config.yaml>` | SFT variant that mixes teacher data (mode B) with the student's own pre-generated outputs (mode A) under tagged system prompts; evals run twice (once per mode). |
| `python scripts/generate_training_data.py <teacher-model>` | Run Alpaca prompts through a teacher and write `stored_outputs/<safe-model-name>.json`. Drops responses that didn't end with the family's EOS token (assumed truncated). |
| `python scripts/sweep.py [--dry-run]` | Cartesian-product launcher: edits-in-place `BASE_CONFIG`/`SWEEP_AXES` in the file, writes one `runs/<sweep>/<run>/config.yaml` per combo, launches each in its own tmux window. `sweep_self_distill.py` is the self-distill variant. |
| `python scripts/rebuild_metadata.py` | One-off recovery script for a specific clobbered sweep — recomputes `metadata.json` from the per-eval JSONs. Hard-coded paths; not a general utility. |

Example configs to copy from: [configs/config.yaml](configs/config.yaml) (SFT) and [configs/rl_config.yaml](configs/rl_config.yaml) (GRPO).

## Architecture

### Config → save_dir contract
[scripts/train.py](scripts/train.py) and [scripts/rl_train.py](scripts/rl_train.py) **require the config file to live inside `save_dir`** (they raise otherwise). The sweep launcher relies on this: it copies the base config into `runs/<sweep>/<run>/config.yaml` with `save_dir` set to that run dir, then trains in place. When writing a new config by hand, put it in `runs/.../config.yaml` (or wherever you want output) and point `save_dir` at the same directory.

### `utils.py` is the core
Almost everything non-trivial lives in [utils.py](utils.py):
- [`GenerateConfig`](utils.py) / [`TrainConfig`](utils.py) / [`SFTExample`](utils.py) — dataclasses; configs map onto these via `filter_dataclass_kwargs` (None → fall back to default).
- [`generate_async`](utils.py) / [`generate_logprobs_async`](utils.py) — semaphore-limited concurrent sampling against a `tinker.SamplingClient`, with a SHA-keyed disk cache at [.generation_cache/](.generation_cache/). Cache key = `(model_id, messages, max_tokens, temperature, num_samples)`. Writes immediately after each generation.
- [`sft_train`](utils.py) — vanilla SFT loop with epoch- or step-based checkpointing. Saves `<run_name>_epoch_N` or `<run_name>_step_N` sampler checkpoints; the trailing integer is parsed back out for the x-axis in plots.
- [`weird_sft_train`](utils.py) — experimental SFT variant whose loss is `(initial_loss − ce_loss)²`, using `forward_backward_custom` for true gradients. Used in self-distillation experiments.
- [`rl_train`](utils.py) — GRPO with reward centering. `value_fn(sampling_client, completion, data_item) → float` is dataset-specific (defined in `scripts/rl_train.py` for Olympiads/APPS).

### Renderer dispatch (model-family-aware)
[`RENDERER_MAP`](utils.py) maps a substring of `tokenizer.name_or_path` (lowercased, **most-specific first**) to a tinker-cookbook renderer name. [`build_generation_input`](utils.py) bypasses the tokenizer's chat template for Llama and DeepSeek (the `RENDERER_GENERATION_FAMILIES` tuple) and goes through the tinker renderer instead — Llama needs knowledge-cutoff handling, DeepSeek needs thinking disabled. **When adding support for a new model family, update both `RENDERER_MAP` and possibly `RENDERER_GENERATION_FAMILIES`**, and add the family's EOS marker to `EOS_TOKENS` in [scripts/train.py](scripts/train.py) and `EOS_TOKENS_BY_FAMILY` in [scripts/generate_training_data.py](scripts/generate_training_data.py).

### Evals
Each eval in [evals/](evals/) (`mmlu.py`, `eval_if.py`, `math_500.py`, `olympiads.py`, `apps.py`, `alpaca.py`) exposes a `run_<name>_evaluation(service_client, paths, system_prompt, config, num_problems, ...)` returning `(scores_per_path, results_per_path)`. Train scripts call them in series via `run_evals_across_paths`; setting an eval's `*_num_problems` to `null` skips it. APPS-only: `apps_test_timeout` controls per-test-case execution timeout when scoring code rollouts.

### Output layout for a run
```
runs/<sweep>/<run>/
  config.yaml
  losses.json           training_data.json
  metadata.json         summary.png
  mmlu/mmlu_<ckpt>.json   ifeval/ifeval_<ckpt>.json
  math_500/...           olympiads/...
```
`metadata.json` carries the resolved config plus per-eval score lists ordered to match `sampling_paths`.

## Conventions worth knowing

- The repo's `.gitignore` excludes `*.json` — committed JSON inputs (`evals/apps_mask.txt`, prompt files) deliberately use `.txt` even when they parse as JSON via `ast.literal_eval`.
- Training prefix prompts can be either `.txt` (wrapped as a single system message) or `.json` (treated verbatim as the chat-message prefix, e.g. system + few-shot exchange) — see [`load_train_prompt_messages`](scripts/train.py).
- Stored teacher outputs in [stored_outputs/](stored_outputs/) are filename-keyed by `model.replace("/", "_")`.
- The codebase was written assuming Qwen3 Instruct family; other families work but require renderer-map entries.
