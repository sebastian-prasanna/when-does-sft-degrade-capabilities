"""Rebuild metadata.json for runs whose original metadata was clobbered.

Recomputes per-checkpoint eval scores from the raw eval JSONs in each
<eval_name>/ subdir. Skips starting_perplexity and sampling_paths/training_paths
(those are not recoverable from disk).
"""

import json
import re
from pathlib import Path
from statistics import mean

import yaml

RUN_DIR = Path('/workspace/when-does-sft-degrade-capabilities/runs/sweep_1')
STUDENTS = ['qwen4b', 'qwen30b', 'qwen235b', 'llama8b', 'llama70b', 'deepseek']
TEACHERS = ['gpt_4.1_nano', 'claude_haiku_4.5', 'gemma_4_26b_a4b_it']
EVALS = ['ifeval', 'math_500', 'mmlu', 'olympiads']

NOTE = (
    "metadata.json was accidentally overwritten by a buggy notebook cell on "
    "2026-05-06; this file was rebuilt from the per-eval JSONs. "
    "sampling_paths, training_paths, and starting_perplexity are not recoverable."
)

CKPT_RE = re.compile(r'_(epoch_\d+|step_\d+)\.json$')


def ckpt_sort_key(name: str) -> tuple[int, int]:
    # epoch_0 first, then step_20, step_40, ...
    m = CKPT_RE.search(name)
    tag = m.group(1) if m else name
    if tag.startswith('epoch_'):
        return (0, int(tag.split('_')[1]))
    return (1, int(tag.split('_')[1]))


def score_file(path: Path, eval_name: str) -> float:
    data = json.loads(path.read_text())
    field = 'score' if eval_name == 'ifeval' else 'correct'
    vals = [float(d[field]) for d in data]
    return mean(vals)


def rebuild(run_dir: Path) -> dict:
    config = yaml.safe_load((run_dir / 'config.yaml').read_text())
    losses = json.loads((run_dir / 'losses.json').read_text())

    metadata: dict = {
        'note': NOTE,
        'config': config,
        'config_path': str(run_dir / 'config.yaml'),
        'num_steps': len(losses),
        'avg_loss': mean(losses),
    }
    for ev in EVALS:
        files = sorted((run_dir / ev).glob(f'{ev}_*.json'), key=lambda p: ckpt_sort_key(p.name))
        metadata[ev] = [score_file(f, ev) for f in files]
    return metadata


def main() -> None:
    for student in STUDENTS:
        for teacher in TEACHERS:
            run_dir = RUN_DIR / f'student_{student}_teacher_{teacher}'
            if not run_dir.is_dir():
                print(f'skip (missing): {run_dir.name}')
                continue
            metadata = rebuild(run_dir)
            (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
            scores = {ev: [round(s, 3) for s in metadata[ev]] for ev in EVALS}
            print(f'{run_dir.name}: {scores}')


if __name__ == '__main__':
    main()
