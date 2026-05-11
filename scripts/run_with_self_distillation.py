#!/usr/bin/env python
"""Self-distillation mixed-mode SFT.

Like scripts/train.py, but mixes the configured training data (mode B) with
the student's own pre-generated outputs (mode A) under tagged system prompts,
trains on the combined 2x dataset, then runs every eval twice — once for each
mode — and plots mode A as dotted lines and mode B as bold lines.

Usage: python scripts/run_with_self_distillation.py <config.yaml>
"""
import argparse
import asyncio
import copy
import json
import math
import random
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
from scripts.train import (  # noqa: E402
    EOS_TOKENS,
    _resolve,
    build_sft_examples,
    filter_dataclass_kwargs,
    load_config,
    load_train_prompt_messages,
    remove_eos_token,
    run_evals_across_paths,
)


# Map a student model id to the JSON file of its own pre-generated outputs.
# These are used as the "mode A" (self-distillation) half of the training mix.
SELF_DISTILL_DIR = REPO_ROOT / "stored_outputs" / "temp_1"
STUDENT_TO_SELF_DISTILL: Dict[str, Path] = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": SELF_DISTILL_DIR / "Qwen_Qwen3-235B-A22B-Instruct-2507.json",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": SELF_DISTILL_DIR / "Qwen_Qwen3-30B-A3B-Instruct-2507.json",
    "Qwen/Qwen3-4B-Instruct-2507": SELF_DISTILL_DIR / "Qwen_Qwen3-4B-Instruct-2507.json",
    "anthropic/claude-haiku-4.5": SELF_DISTILL_DIR / "anthropic_claude-haiku-4.5.json",
    "deepseek-ai/DeepSeek-V3.1": SELF_DISTILL_DIR / "deepseek-ai_DeepSeek-V3.1.json",
    "google/gemma-4-26b-a4b-it": SELF_DISTILL_DIR / "google_gemma-4-26b-a4b-it.json",
    "meta-llama/Llama-3.1-8B-Instruct": SELF_DISTILL_DIR / "meta-llama_Llama-3.1-8B-Instruct.json",
    "meta-llama/Llama-3.3-70B-Instruct": SELF_DISTILL_DIR / "meta-llama_Llama-3.3-70B-Instruct.json",
    "openai/gpt-4.1-nano": SELF_DISTILL_DIR / "openai_gpt-4.1-nano.json",
    "openai/gpt-4o-mini": SELF_DISTILL_DIR / "openai_gpt-4o-mini.json",
}


def add_mode_tag_to_messages(
    prefix_messages: List[Dict[str, str]], mode: str
) -> List[Dict[str, str]]:
    """Append a mode tag to the first system message in a prefix.

    The training prefix may be a single system message (loaded from a .txt) or
    a system + few-shot exchange (loaded from a .json). Either way the very
    first message must be the system message — that's what we tag.
    """
    if not prefix_messages or prefix_messages[0].get("role") != "system":
        raise ValueError(
            "Training prefix must start with a system message so we can attach "
            f"the mode tag. Got: {prefix_messages[:1]}"
        )
    tagged = copy.deepcopy(prefix_messages)
    tagged[0]["content"] = f"{tagged[0]['content']} <mode>{mode}</mode>"
    return tagged


def add_mode_tag_to_text(system_prompt: str, mode: str) -> str:
    return f"{system_prompt} <mode>{mode}</mode>"


def make_summary_plot(
    losses: List[float],
    checkpoint_names: List[str],
    scores_by_path_a: List[Dict[str, float]],
    scores_by_path_b: List[Dict[str, float]],
    n_per_eval: Dict[str, int],
    save_path: Path,
    title: str,
) -> None:
    set_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(range(1, len(losses) + 1), losses)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")

    steps = [int(name.split("_")[-1]) for name in checkpoint_names]

    eval_names = sorted(
        {n for d in scores_by_path_a for n in d}
        | {n for d in scores_by_path_b for n in d}
    )

    if eval_names and checkpoint_names:
        # One color per eval, A=dotted, B=bold solid.
        cmap = plt.rcParams["axes.prop_cycle"].by_key().get("color")
        for idx, ev in enumerate(eval_names):
            color = cmap[idx % len(cmap)] if cmap else None
            n = n_per_eval.get(ev, 0)
            for scores_by_path, mode, linestyle, linewidth in [
                (scores_by_path_a, "A", ":", 1.5),
                (scores_by_path_b, "B", "-", 2.5),
            ]:
                xs, ys, errs = [], [], []
                for x, d in zip(steps, scores_by_path):
                    if ev in d:
                        p = d[ev]
                        xs.append(x)
                        ys.append(p)
                        errs.append(1.96 * math.sqrt(p * (1 - p) / n) if n > 0 else 0.0)
                if xs:
                    axes[1].errorbar(
                        xs, ys, yerr=errs,
                        marker="o", capsize=3,
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        label=f"{ev} (mode {mode})",
                    )
        axes[1].set_xlabel("Gradient steps")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Score")
        axes[1].set_title("Eval scores by gradient step (95% CI) — mode A dotted, mode B bold")
        axes[1].legend(fontsize=8)
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

    student_model = config["student_model"]
    if student_model not in STUDENT_TO_SELF_DISTILL:
        raise KeyError(
            f"No self-distillation dataset registered for student_model={student_model!r}. "
            f"Add an entry to STUDENT_TO_SELF_DISTILL in {Path(__file__).name}."
        )
    self_distill_path = STUDENT_TO_SELF_DISTILL[student_model]
    if not self_distill_path.exists():
        raise FileNotFoundError(f"Self-distillation file missing: {self_distill_path}")

    train_prefix_messages = load_train_prompt_messages(_resolve(config["train_system_prompt_file"]))
    eval_system_prompt = _resolve(config["eval_system_prompt_file"]).read_text().strip()

    train_cfg_kwargs = filter_dataclass_kwargs(TrainConfig, config.get("train_config") or {})
    train_cfg = TrainConfig(**train_cfg_kwargs)

    gen_cfg_kwargs = filter_dataclass_kwargs(GenerateConfig, config.get("generate_config") or {})
    gen_cfg = GenerateConfig(**gen_cfg_kwargs)

    prefix_a = add_mode_tag_to_messages(train_prefix_messages, "A")
    prefix_b = add_mode_tag_to_messages(train_prefix_messages, "B")

    examples_a = build_sft_examples(self_distill_path, prefix_a, train_cfg.num_examples)
    examples_b = build_sft_examples(_resolve(config["training_data"]), prefix_b, train_cfg.num_examples)
    print(
        f"Loaded {len(examples_a)} self-distill examples (mode A) from {self_distill_path}\n"
        f"Loaded {len(examples_b)} normal examples       (mode B) from {config['training_data']}"
    )

    examples = examples_a + examples_b
    random.seed(42)
    random.shuffle(examples)
    print(f"Combined and shuffled to {len(examples)} total training examples.")

    service_client = tinker.ServiceClient()
    training_path = config.get("training_path")
    if training_path:
        print(f"Loading starting weights from: {training_path}")
        training_client = service_client.create_training_client_from_state(training_path)
    else:
        training_client = service_client.create_lora_training_client(base_model=student_model)

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

    eval_prompt_a = add_mode_tag_to_text(eval_system_prompt, "A")
    eval_prompt_b = add_mode_tag_to_text(eval_system_prompt, "B")

    scores_a, results_a = await run_evals_across_paths(
        service_client=service_client,
        paths=sampling_paths,
        system_prompt=eval_prompt_a,
        eval_cfg=config.get("eval") or {},
        gen_cfg=gen_cfg,
    )
    scores_b, results_b = await run_evals_across_paths(
        service_client=service_client,
        paths=sampling_paths,
        system_prompt=eval_prompt_b,
        eval_cfg=config.get("eval") or {},
        gen_cfg=gen_cfg,
    )

    # Layout: <eval_name>/<mode>/<eval_name>_<ckpt>.json
    for mode, results_by_path in [("A", results_a), ("B", results_b)]:
        for ckpt_name, results in zip(checkpoint_names, results_by_path):
            for eval_name, eval_result in results.items():
                eval_dir = save_dir / eval_name / f"mode_{mode}"
                eval_dir.mkdir(parents=True, exist_ok=True)
                with open(eval_dir / f"{eval_name}_{ckpt_name}.json", "w") as f:
                    json.dump(eval_result, f, indent=2)

    metadata: Dict[str, Any] = {
        "config": config,
        "config_path": str(config_path),
        "self_distillation_data": str(self_distill_path),
        "num_steps": train_result["num_steps"],
        "avg_loss": train_result["avg_loss"],
        "sampling_paths": sampling_paths,
        "training_paths": train_result["training_paths"],
    }
    all_eval_names = sorted({n for d in scores_a + scores_b for n in d})
    for ev in all_eval_names:
        metadata[f"{ev}_mode_A"] = [d.get(ev) for d in scores_a]
        metadata[f"{ev}_mode_B"] = [d.get(ev) for d in scores_b]
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    first_results = next((r for r in (results_a + results_b) if r), {})
    n_per_eval = {ev: len(res) for ev, res in first_results.items()}

    plot_title = (
        f"{student_model} self-distillation "
        f"({Path(config['training_data']).stem})"
    )
    make_summary_plot(
        losses=train_result["losses"],
        checkpoint_names=checkpoint_names,
        scores_by_path_a=scores_a,
        scores_by_path_b=scores_b,
        n_per_eval=n_per_eval,
        save_path=save_dir / "summary.png",
        title=plot_title,
    )

    print("\n=== Done ===")
    print(f"save_dir: {save_dir}")
    for name, sa, sb in zip(checkpoint_names, scores_a, scores_b):
        print(f"  {name}: mode_A={sa}  mode_B={sb}")


if __name__ == "__main__":
    asyncio.run(main())
