"""Sweep launcher: GRPO from each SFT checkpoint in runs/rerun_for_grpo,
on both olympiads and apps. 6 checkpoints x 2 datasets = 12 runs.

Usage:
    1. Edit BASE_CONFIG, SWEEP_NAME and SWEEP_AXES below.
    2. Run: python runs/grpo_from_sft/sweep.py [--dry-run]

Each axis is a list of dicts whose keys are config.yaml keys. Use
dot-separated keys to override nested values (e.g. "rl_config.dataset").
The `_label` key is special: it is stripped from the merged config and
joined across axes to form the run name.

The cartesian product across all axes determines the runs. A single-element
axis acts as a global override applied to every run.

For each combination, we write runs/<SWEEP_NAME>/<run_name>/config.yaml
(with save_dir set to the run dir so rl_train.py's "config inside save_dir"
check passes) and launch a tmux window running scripts/rl_train.py on it.
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Configuration — edit these for each sweep.
# ---------------------------------------------------------------------------

BASE_CONFIG = "configs/rl_config.yaml"
SWEEP_NAME = "grpo_from_sft"

SWEEP_AXES: list[list[dict[str, Any]]] = [
    # Axis 1: SFT checkpoint to start GRPO from. student_model and
    # starting_training_path are linked — must come from the same SFT run.
    [
        {
            "_label": "student_qwen30b",
            "student_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "starting_training_path": "tinker://105c8edf-8e8e-5c5e-86fc-492b13e72f07:train:0/weights/student_qwen30b_teacher_llama8b_epoch_1",
        }
    ],
    # Axis 2: RL dataset. train_split must match the dataset (default is "red"
    # for olympiads, "apps" for apps).
    [
        {"_label": "olympiads", "rl_config.dataset": "olympiads", "rl_config.train_split": "red"},
        {"_label": "apps",      "rl_config.dataset": "apps",      "rl_config.train_split": "apps"},
    ],
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
        # save_dir is what rl_train.py reads; set it relative to repo root so
        # the run is reproducible across machines.
        cfg["save_dir"] = str(run_dir.relative_to(REPO_ROOT))

        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        config_paths.append(config_path)

    return config_paths


def launch_tmux(session_name: str, config_paths: list[Path], delay: int = 0) -> None:
    session = tmux_safe(session_name)
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)

    python_bin = sys.executable
    train_script = REPO_ROOT / "scripts" / "rl_train.py"

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
    parser = argparse.ArgumentParser(description="Generate sweep configs and launch in tmux.")
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
