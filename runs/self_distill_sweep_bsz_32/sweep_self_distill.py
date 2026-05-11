"""Sweep launcher for run_with_self_distillation.py.

Same idea as scripts/sweep.py — cartesian product of config overrides, one
tmux window per combination — but each window runs
scripts/run_with_self_distillation.py instead of scripts/train.py.

Usage:
    1. Edit BASE_CONFIG, SWEEP_NAME and SWEEP_AXES below.
    2. Run: python scripts/sweep_self_distill.py [--dry-run]
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` until we find the repo root (a `scripts/` sibling of `configs/`).

    Lets this script run from anywhere in the tree (e.g. a copy alongside its
    sweep dir under runs/<sweep_name>/) without a hard-coded depth.
    """
    for parent in (start, *start.parents):
        if (parent / "scripts").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repo root from {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())

# ---------------------------------------------------------------------------
# Configuration — edit these for each sweep.
# ---------------------------------------------------------------------------

BASE_CONFIG = "configs/config.yaml"
SWEEP_NAME = "self_distill_sweep_bsz_32"

SWEEP_AXES: list[list[dict[str, Any]]] = [
    # Axis 1: student
    [
        {"_label": "student_qwen4b",     "student_model": "Qwen/Qwen3-4B-Instruct-2507"},
        {"_label": "student_qwen30b",    "student_model": "Qwen/Qwen3-30B-A3B-Instruct-2507"},
        {"_label": "student_qwen235b",   "student_model": "Qwen/Qwen3-235B-A22B-Instruct-2507"},
        {"_label": "student_llama8b",    "student_model": "meta-llama/Llama-3.1-8B-Instruct"},
        {"_label": "student_llama70b",   "student_model": "meta-llama/Llama-3.3-70B-Instruct"},
        {"_label": "student_deepseek",   "student_model": "deepseek-ai/DeepSeek-V3.1"},
    ],
    # Axis 2: student model (single-element axis = global override).
    [
        {
            "_label": "teacher_llama_8b",     
            "training_data": "stored_outputs/temp_1/meta-llama_Llama-3.1-8B-Instruct.json",
            "train_config.batch_size": 32, # need to fix the number of steps
        }
    ]
]

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

TMUX_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def tmux_safe(name: str) -> str:
    """Sanitize a string for tmux session/window names (tmux rejects '.', etc.)."""
    return TMUX_NAME_RE.sub("_", name)


def deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set d['a']['b']['c'] = value when dotted_key == 'a.b.c'."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def make_run_name(combo: tuple[dict, ...]) -> str:
    labels = [d["_label"] for d in combo if "_label" in d]
    return "_".join(labels) if labels else "run"


def generate_configs(
    base_path: Path,
    sweep_name: str,
    axes: list[list[dict]],
) -> list[Path]:
    base_cfg = yaml.safe_load(base_path.read_text())
    combos = list(itertools.product(*axes))
    config_paths: list[Path] = []
    seen_names: set[str] = set()

    for idx, combo in enumerate(combos):
        cfg = deepcopy(base_cfg)
        for d in combo:
            for k, v in d.items():
                if k == "_label":
                    continue
                deep_set(cfg, k, v)

        run_name = make_run_name(combo)
        if run_name in seen_names:
            run_name = f"{run_name}_{idx}"
        seen_names.add(run_name)

        run_dir = REPO_ROOT / "runs" / sweep_name / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg["save_dir"] = str(run_dir.relative_to(REPO_ROOT))

        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        config_paths.append(config_path)

    return config_paths


def launch_tmux(session_name: str, config_paths: list[Path], delay: int = 0) -> None:
    session = tmux_safe(session_name)
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)

    python_bin = sys.executable
    train_script = REPO_ROOT / "scripts" / "run_with_self_distillation.py"

    for i, config_path in enumerate(config_paths):
        window = tmux_safe(config_path.parent.name)
        sleep_prefix = f"sleep {i * delay} && " if delay > 0 and i > 0 else ""
        cmd = f"cd {REPO_ROOT} && {sleep_prefix}{python_bin} {train_script} {config_path}"
        if i == 0:
            subprocess.run([
                "tmux", "new-session", "-d", "-s", session, "-n", window, "-c", str(REPO_ROOT),
            ], check=True)
        else:
            subprocess.run([
                "tmux", "new-window", "-t", session, "-n", window, "-c", str(REPO_ROOT),
            ], check=True)
        subprocess.run([
            "tmux", "send-keys", "-t", f"{session}:{window}", cmd, "Enter",
        ], check=True)

    print(f"\nLaunched {len(config_paths)} windows in tmux session '{session}'.")
    print(f"Attach with: tmux attach -t {session}")
    print("Navigate: Ctrl-B n (next), Ctrl-B p (prev), Ctrl-B w (list).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-distill sweep configs and launch in tmux.")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't launch tmux.")
    parser.add_argument("--delay", type=int, default=300,
                        help="Stagger run i by i*delay seconds (default 300; 0 disables).")
    args = parser.parse_args()

    total = 1
    for axis in SWEEP_AXES:
        total *= len(axis)

    print(f"Sweep '{SWEEP_NAME}': {len(SWEEP_AXES)} axes, {total} combinations")
    for i, axis in enumerate(SWEEP_AXES):
        keys = sorted({k for d in axis for k in d if k != "_label"})
        print(f"  Axis {i}: {len(axis)} values, keys={keys}")

    base_path = REPO_ROOT / BASE_CONFIG
    if not base_path.exists():
        sys.exit(f"BASE_CONFIG not found: {base_path}")

    config_paths = generate_configs(base_path, SWEEP_NAME, SWEEP_AXES)
    print(f"\nGenerated {len(config_paths)} configs:")
    for p in config_paths:
        print(f"  {p.relative_to(REPO_ROOT)}")

    if args.dry_run:
        print("\n(dry run — not launching)")
        return

    launch_tmux(SWEEP_NAME, config_paths, delay=args.delay)


if __name__ == "__main__":
    main()
