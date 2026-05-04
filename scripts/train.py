#!/usr/bin/env python
"""SFT-train a student model on stored teacher outputs, evaluate, and plot.

Usage: python scripts/train.py <config.yaml>

The config drives every step:
  1. SFT-train the student model on (problem, response) pairs from a JSON
     file in stored_outputs/.
  2. Evaluate the trained student on MMLU, IFEval, and MATH-500.
  3. Save metadata, losses, training data, eval logs, and a summary plot
     under save_dir.
"""
import argparse
import asyncio
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))

import tinker  # noqa: E402

from utils import (  # noqa: E402
    GenerateConfig,
    SFTExample,
    TrainConfig,
    set_matplotlib_style,
    sft_train,
)
from evals.eval_if import run_ifeval_evaluation  # noqa: E402
from evals.math_500 import run_math_500_evaluation  # noqa: E402
from evals.mmlu import run_mmlu_evaluation  # noqa: E402
from evals.olympiads import run_olympiads_evaluation  # noqa: E402


def _resolve(path: str) -> Path:
    """Resolve a config path relative to the repo root unless already absolute."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# Trailing EOS markers seen across teacher outputs in stored_outputs/.
EOS_TOKENS = (
    "<|im_end|>",            # Qwen
    "<|eot_id|>",            # Llama 3
    "<｜end▁of▁sentence｜>",  # DeepSeek
    "<|endoftext|>",
    "<|end|>",
)


def remove_eos_token(text: str) -> str:
    """Strip a trailing EOS marker (and surrounding whitespace) from a response."""
    text = text.rstrip()
    for tok in EOS_TOKENS:
        if text.endswith(tok):
            return text[: -len(tok)].rstrip()
    return text


def load_train_prompt_messages(path: Path) -> List[Dict[str, str]]:
    """Load the training-prompt prefix as a list of chat messages.

    .json: parsed as a list of chat messages used verbatim as the SFTExample prefix.
    Anything else: treated as a text file and wrapped in a single system message.
    """
    if path.suffix.lower() == ".json":
        with open(path) as f:
            messages = json.load(f)
        if not isinstance(messages, list):
            raise ValueError(f"{path} must contain a JSON list of chat messages")
        return messages
    return [{"role": "system", "content": path.read_text().strip()}]


def build_sft_examples(
    json_path: Path, prefix_messages: List[Dict[str, str]], num_examples: int
) -> List[SFTExample]:
    """Convert (problem, response) entries into SFTExamples with a fixed message prefix."""
    with open(json_path) as f:
        data = json.load(f)
    data = data[:num_examples]
    return [
        SFTExample(
            input=[
                *prefix_messages,
                {"role": "user", "content": entry["problem"]},
            ],
            output=[{"role": "assistant", "content": remove_eos_token(entry["response"])}],
        )
        for entry in data
    ]


def filter_dataclass_kwargs(cls, src: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only known dataclass fields; drop None so dataclass defaults apply."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in src.items() if k in valid and v is not None}


async def run_evals_across_paths(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    eval_cfg: Dict[str, Any],
    gen_cfg: GenerateConfig,
) -> Tuple[List[Dict[str, float]], List[Dict[str, list]]]:
    """Run mmlu / math_500 / ifeval across all sampling paths in parallel.

    Returns (scores_by_path, results_by_path), each a list aligned with `paths`.
    Each entry maps eval_name -> score (or per-problem result list).
    """
    scores_by_path: List[Dict[str, float]] = [{} for _ in paths]
    results_by_path: List[Dict[str, list]] = [{} for _ in paths]

    if eval_cfg.get("mmlu_num_problems") is not None:
        accs, all_results = await run_mmlu_evaluation(
            service_client=service_client,
            paths=paths,
            system_prompt=system_prompt,
            config=gen_cfg,
            num_problems=eval_cfg["mmlu_num_problems"],
            save=False,
        )
        for i, (acc, res) in enumerate(zip(accs, all_results)):
            scores_by_path[i]["mmlu"] = float(acc)
            results_by_path[i]["mmlu"] = res

    if "math500_num_problems" in eval_cfg:
        accs, all_results = await run_math_500_evaluation(
            service_client=service_client,
            paths=paths,
            system_prompt=system_prompt,
            config=gen_cfg,
            num_problems=eval_cfg["math500_num_problems"],
            save=False,
        )
        for i, (acc, res) in enumerate(zip(accs, all_results)):
            scores_by_path[i]["math_500"] = float(acc)
            results_by_path[i]["math_500"] = res

    if eval_cfg.get("olympiads_num_problems") is not None:
        olym_kwargs = {}
        if eval_cfg.get("olympiads_split"):
            olym_kwargs["split"] = eval_cfg["olympiads_split"]
        accs, all_results = await run_olympiads_evaluation(
            service_client=service_client,
            paths=paths,
            system_prompt=system_prompt,
            config=gen_cfg,
            num_problems=eval_cfg["olympiads_num_problems"],
            save=False,
            **olym_kwargs,
        )
        for i, (acc, res) in enumerate(zip(accs, all_results)):
            scores_by_path[i]["olympiads"] = float(acc)
            results_by_path[i]["olympiads"] = res

    if eval_cfg.get("ifeval_num_problems") is not None:
        if_kwargs = {}
        if eval_cfg.get("judge_model"):
            if_kwargs["judge_model"] = eval_cfg["judge_model"]
        avg_scores, all_results = await run_ifeval_evaluation(
            service_client=service_client,
            paths=paths,
            system_prompt=system_prompt,
            config=gen_cfg,
            num_problems=eval_cfg["ifeval_num_problems"],
            save=False,
            **if_kwargs,
        )
        for i, (s, res) in enumerate(zip(avg_scores, all_results)):
            scores_by_path[i]["ifeval"] = float(s)
            results_by_path[i]["ifeval"] = res

    return scores_by_path, results_by_path


def make_summary_plot(
    losses: List[float],
    checkpoint_names: List[str],
    scores_by_path: List[Dict[str, float]],
    n_per_eval: Dict[str, int],
    save_path: Path,
    title: str,
) -> None:
    import math

    set_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(range(1, len(losses) + 1), losses)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")

    # x-axis = gradient steps, parsed as the trailing integer of the checkpoint name
    # (e.g. "..._step_30" -> 30, "..._epoch_0" -> 0).
    steps = [int(name.split("_")[-1]) for name in checkpoint_names]

    eval_names = sorted({n for d in scores_by_path for n in d})
    if eval_names and checkpoint_names:
        for ev in eval_names:
            xs, ys, errs = [], [], []
            n = n_per_eval.get(ev, 0)
            for x, d in zip(steps, scores_by_path):
                if ev in d:
                    p = d[ev]
                    xs.append(x)
                    ys.append(p)
                    errs.append(1.96 * math.sqrt(p * (1 - p) / n) if n > 0 else 0.0)
            axes[1].errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=ev)
        axes[1].set_xlabel("Gradient steps")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Score")
        axes[1].set_title("Eval scores by gradient step (95% CI)")
        axes[1].legend()
    else:
        axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


async def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a config.yaml file")
    args = parser.parse_args(argv)

    config_path = _resolve(args.config).resolve()
    config = load_config(config_path)

    save_dir = _resolve(config["save_dir"]).resolve()
    if not config_path.is_relative_to(save_dir):
        raise ValueError(
            "Config file must live inside save_dir.\n"
            f"  config_path: {config_path}\n"
            f"  save_dir:    {save_dir}"
        )
    save_dir.mkdir(parents=True, exist_ok=True)

    train_prefix_messages = load_train_prompt_messages(_resolve(config["train_system_prompt_file"]))
    eval_system_prompt = _resolve(config["eval_system_prompt_file"]).read_text().strip()

    train_cfg_kwargs = filter_dataclass_kwargs(TrainConfig, config.get("train_config") or {})
    train_cfg = TrainConfig(**train_cfg_kwargs)

    gen_cfg_kwargs = filter_dataclass_kwargs(GenerateConfig, config.get("generate_config") or {})
    gen_cfg = GenerateConfig(**gen_cfg_kwargs)

    examples = build_sft_examples(_resolve(config["training_data"]), train_prefix_messages, train_cfg.num_examples)
    print(f"Loaded {len(examples)} SFT examples from {config['training_data']}")

    service_client = tinker.ServiceClient()
    training_path = config.get("training_path")
    if training_path:
        print(f"Loading starting weights from: {training_path}")
        training_client = service_client.create_training_client_from_state(training_path)
    else:
        training_client = service_client.create_lora_training_client(base_model=config["student_model"])

    run_name = save_dir.name
    train_result = sft_train(
        training_client=training_client,
        data=examples,
        config=train_cfg,
        run_name=run_name,
    )

    with open(save_dir / "losses.json", "w") as f:
        json.dump(train_result["losses"], f, indent=2)
    with open(save_dir / "training_data.json", "w") as f:
        json.dump(train_result["training_data"], f, indent=2)

    sampling_paths = train_result["sampling_paths"]
    checkpoint_names = [p.split("/")[-1] for p in sampling_paths]
    print(f"Evaluating across {len(sampling_paths)} checkpoint(s): {checkpoint_names}")

    scores_by_path, results_by_path = await run_evals_across_paths(
        service_client=service_client,
        paths=sampling_paths,
        system_prompt=eval_system_prompt,
        eval_cfg=config.get("eval") or {},
        gen_cfg=gen_cfg,
    )

    # Layout: one folder per eval, one file per checkpoint inside.
    for ckpt_name, results in zip(checkpoint_names, results_by_path):
        for eval_name, eval_result in results.items():
            eval_dir = save_dir / eval_name
            eval_dir.mkdir(parents=True, exist_ok=True)
            with open(eval_dir / f"{eval_name}_{ckpt_name}.json", "w") as f:
                json.dump(eval_result, f, indent=2)

    metadata = {
        "config": config,
        "config_path": str(config_path),
        "num_steps": train_result["num_steps"],
        "avg_loss": train_result["avg_loss"],
        "sampling_paths": sampling_paths,
        "training_paths": train_result["training_paths"],
    }
    # Eval scores as flat lists, ordered to match sampling_paths.
    for ev in sorted({n for d in scores_by_path for n in d}):
        metadata[ev] = [d.get(ev) for d in scores_by_path]
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    n_per_eval = {ev: len(res) for ev, res in (results_by_path[0] if results_by_path else {}).items()}

    plot_title = f"{config['student_model']}  ({Path(config['training_data']).stem})"
    make_summary_plot(
        losses=train_result["losses"],
        checkpoint_names=checkpoint_names,
        scores_by_path=scores_by_path,
        n_per_eval=n_per_eval,
        save_path=save_dir / "summary.png",
        title=plot_title,
    )

    print("\n=== Done ===")
    print(f"save_dir: {save_dir}")
    for name, scores in zip(checkpoint_names, scores_by_path):
        print(f"  {name}: {scores}")


if __name__ == "__main__":
    asyncio.run(main())
