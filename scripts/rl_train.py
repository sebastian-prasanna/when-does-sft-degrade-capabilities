#!/usr/bin/env python
"""GRPO-train a model on Olympiads, evaluate, and plot.

Usage: python scripts/rl_train.py <rl_config.yaml>

Mirrors scripts/train.py but uses utils.rl_train instead of utils.sft_train:
  1. Optionally load weights from a starting tinker:// path (a SFT checkpoint).
  2. Loop GRPO iterations: sample a batch of olympiad problems, generate
     `group_size` completions per problem, score with the answer-tag value
     function, take a policy-gradient step.
  3. Evaluate the saved sampling checkpoints on the configured eval suite.
  4. Save metadata, rewards, eval logs, and a summary plot under save_dir.
"""
import argparse
import asyncio
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
    extract_xml_tag,
    rl_train,
    set_matplotlib_style,
)
from evals.eval_if import run_ifeval_evaluation  # noqa: E402
from evals.math_500 import run_math_500_evaluation  # noqa: E402
from evals.mmlu import run_mmlu_evaluation  # noqa: E402
from evals.olympiads import (  # noqa: E402
    OLYMPIADS_PROMPT_PATH,
    load_olympiads_dataset,
    run_olympiads_evaluation,
)


def _resolve(path: str) -> Path:
    """Resolve a config path relative to the repo root unless already absolute."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def filter_dataclass_kwargs(cls, src: Dict[str, Any]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in src.items() if k in valid and v is not None}


def make_olympiads_format_fn(system_prompt: str, olympiads_prompt: str):
    """Returns a format_fn(data_item) -> chat messages for utils.rl_train."""
    def format_fn(data_item: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": olympiads_prompt.format(problem_statement=data_item["problem"])},
        ]
    return format_fn


def olympiads_value_fn(sampling_client, completion: str, data_item: Dict[str, Any]) -> float:
    """Reward = 1.0 iff <answer>...</answer> matches the ground-truth answer."""
    predicted = extract_xml_tag(completion, "answer")
    if predicted is not None:
        predicted = predicted.strip()
    expected = data_item["answer"].strip()
    return 1.0 if predicted == expected else 0.0


async def run_evals_across_paths(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    eval_cfg: Dict[str, Any],
    gen_cfg: GenerateConfig,
) -> Tuple[List[Dict[str, float]], List[Dict[str, list]]]:
    """Run mmlu / math_500 / ifeval / olympiads across all sampling paths."""
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
    avg_rewards: List[float],
    eval_iterations: List[int],
    scores_by_path: List[Dict[str, float]],
    n_per_eval: Dict[str, int],
    save_path: Path,
    title: str,
) -> None:
    set_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(range(1, len(avg_rewards) + 1), avg_rewards, marker="o")
    axes[0].set_xlabel("GRPO iteration")
    axes[0].set_ylabel("Mean training reward")
    axes[0].set_title("Training reward")

    eval_names = sorted({n for d in scores_by_path for n in d})
    if eval_names and eval_iterations:
        for ev in eval_names:
            xs, ys, errs = [], [], []
            n = n_per_eval.get(ev, 0)
            for x, d in zip(eval_iterations, scores_by_path):
                if ev in d:
                    p = d[ev]
                    xs.append(x)
                    ys.append(p)
                    errs.append(1.96 * math.sqrt(p * (1 - p) / n) if n > 0 else 0.0)
            axes[1].errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=ev)
        axes[1].set_xlabel("GRPO iteration")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Score")
        axes[1].set_title("Eval scores by iteration (95% CI)")
        axes[1].legend()
    else:
        axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


async def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a rl_config.yaml file")
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

    train_system_prompt = _resolve(config["train_system_prompt_file"]).read_text().strip()
    eval_system_prompt = _resolve(config["eval_system_prompt_file"]).read_text().strip()
    olympiads_prompt_path = (
        _resolve(config["olympiads_prompt_file"])
        if config.get("olympiads_prompt_file")
        else OLYMPIADS_PROMPT_PATH
    )
    olympiads_prompt = olympiads_prompt_path.read_text()

    rl_cfg = config["rl_config"]
    gen_cfg_kwargs = filter_dataclass_kwargs(GenerateConfig, config.get("generate_config") or {})
    eval_gen_cfg = GenerateConfig(**gen_cfg_kwargs)

    # Sampling config used during RL rollouts. group_size → num_samples.
    rl_gen_cfg = GenerateConfig(
        temperature=rl_cfg.get("temperature", 1.0),
        max_tokens=rl_cfg.get("max_tokens", 4000),
        max_concurrent=rl_cfg.get("max_concurrent", 2000),
        num_samples=rl_cfg["group_size"],
        cache=False,  # RL needs fresh samples each iter
    )

    # Service client + training client (optionally from a starting checkpoint).
    service_client = tinker.ServiceClient()
    starting_path = config.get("starting_training_path")
    if starting_path:
        print(f"Loading starting weights from: {starting_path}")
        training_client = service_client.create_training_client_from_state(starting_path)
    else:
        print(f"Creating fresh LoRA training client for: {config['student_model']}")
        training_client = service_client.create_lora_training_client(base_model=config["student_model"])

    run_name = save_dir.name

    # Save weights at iter 0 so we evaluate the starting policy too.
    initial_path = training_client.save_weights_for_sampler(name=f"{run_name}_iter_0").result().path
    print(f"Saved initial sampling checkpoint: {initial_path}")
    sampling_client = service_client.create_sampling_client(model_path=initial_path)

    # Load training dataset.
    train_split = rl_cfg.get("train_split", "red")
    num_train_problems = rl_cfg.get("num_train_problems")
    full_dataset = load_olympiads_dataset(split=train_split)
    if num_train_problems is not None:
        full_dataset = full_dataset[:num_train_problems]
    print(f"Loaded {len(full_dataset)} olympiad problems from split '{train_split}'")

    format_fn = make_olympiads_format_fn(train_system_prompt, olympiads_prompt)

    rng = random.Random(rl_cfg.get("seed", 42))
    num_iterations = rl_cfg["num_iterations"]
    problems_per_iter = rl_cfg["problems_per_iter"]
    save_every_n_iters = rl_cfg.get("save_every_n_iters", 1)

    def _new_epoch_queue() -> List[int]:
        order = list(range(len(full_dataset)))
        rng.shuffle(order)
        return order

    # Epoch-style sampler: shuffle once, drain by problems_per_iter, reshuffle when
    # not enough left for a full batch. Drops the small remainder at each epoch
    # boundary so every batch stays the same size and contains unique problems.
    index_queue: List[int] = _new_epoch_queue()
    epoch_num = 1

    sampling_paths: List[str] = [initial_path]
    eval_iterations: List[int] = [0]
    avg_rewards: List[float] = []
    iter_metrics: List[Dict[str, Any]] = []

    for it in range(1, num_iterations + 1):
        if len(index_queue) < problems_per_iter:
            index_queue = _new_epoch_queue()
            epoch_num += 1
        print(f"\n=== GRPO iteration {it}/{num_iterations} (epoch {epoch_num}) ===")
        batch_indices = index_queue[:problems_per_iter]
        index_queue = index_queue[problems_per_iter:]
        # Fresh shallow copies so rl_train's in-place 'input_text' tagging doesn't accumulate.
        batch = [dict(full_dataset[i]) for i in batch_indices]

        result = rl_train(
            training_client=training_client,
            sampling_client=sampling_client,
            dataset=batch,
            format_fn=format_fn,
            value_fn=olympiads_value_fn,
            config=rl_gen_cfg,
            learning_rate=rl_cfg["learning_rate"],
            batch_size=rl_cfg.get("batch_size"),
            run_name=f"{run_name}_iter_{it}",
            normalize_advantages_by_length=rl_cfg.get("normalize_advantages_by_length", False),
            center_rewards=rl_cfg.get("center_rewards", True),
        )

        avg_rewards.append(result["avg_reward"])
        iter_metrics.append({
            "iteration": it,
            "avg_reward": result["avg_reward"],
            "num_datums": result["num_datums"],
            "rewards": result["rewards"],
            "optim_metrics": result["optim_metrics"],
        })

        new_path = result["sampling_paths"][-1]
        # Refresh sampling client so the next rollout uses updated weights.
        sampling_client = service_client.create_sampling_client(model_path=new_path)

        if it % save_every_n_iters == 0 or it == num_iterations:
            sampling_paths.append(new_path)
            eval_iterations.append(it)
            print(f"Iter {it}: avg_reward={result['avg_reward']:.4f}  ckpt={new_path}")
        else:
            print(f"Iter {it}: avg_reward={result['avg_reward']:.4f}  (no eval ckpt)")

    # Persist training-time artifacts.
    with open(save_dir / "rewards.json", "w") as f:
        json.dump(avg_rewards, f, indent=2)
    with open(save_dir / "iter_metrics.json", "w") as f:
        json.dump(iter_metrics, f, indent=2)

    checkpoint_names = [p.split("/")[-1] for p in sampling_paths]
    print(f"\nEvaluating across {len(sampling_paths)} checkpoint(s): {checkpoint_names}")

    scores_by_path, results_by_path = await run_evals_across_paths(
        service_client=service_client,
        paths=sampling_paths,
        system_prompt=eval_system_prompt,
        eval_cfg=config.get("eval") or {},
        gen_cfg=eval_gen_cfg,
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
        "starting_training_path": starting_path,
        "num_iterations": num_iterations,
        "avg_rewards": avg_rewards,
        "sampling_paths": sampling_paths,
        "eval_iterations": eval_iterations,
    }
    for ev in sorted({n for d in scores_by_path for n in d}):
        metadata[ev] = [d.get(ev) for d in scores_by_path]
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    n_per_eval = {ev: len(res) for ev, res in (results_by_path[0] if results_by_path else {}).items()}

    plot_title = f"GRPO {config['student_model']} on Olympiads ({train_split})"
    make_summary_plot(
        avg_rewards=avg_rewards,
        eval_iterations=eval_iterations,
        scores_by_path=scores_by_path,
        n_per_eval=n_per_eval,
        save_path=save_dir / "summary.png",
        title=plot_title,
    )

    print("\n=== Done ===")
    print(f"save_dir: {save_dir}")
    for it, name, scores in zip(eval_iterations, checkpoint_names, scores_by_path):
        print(f"  iter {it} ({name}): {scores}")


if __name__ == "__main__":
    asyncio.run(main())
