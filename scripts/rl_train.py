#!/usr/bin/env python
"""GRPO-train a model on Olympiads or APPS, evaluate, and plot.

Usage: python scripts/rl_train.py <rl_config.yaml>

Mirrors scripts/train.py but uses utils.rl_train instead of utils.sft_train:
  1. Optionally load weights from a starting tinker:// path (a SFT checkpoint).
  2. Loop GRPO iterations: sample a batch of problems (olympiads or APPS),
     generate `group_size` completions per problem, score with the
     dataset-specific value function, take a policy-gradient step.
  3. Evaluate the saved sampling checkpoints on the configured eval suite.
  4. Save metadata, rewards, eval logs, and a summary plot under save_dir.
"""
import argparse
import ast
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
from evals.apps import (  # noqa: E402
    APPS_PROMPT_PATH,
    load_apps_dataset,
    run_apps_evaluation,
    test_solutions_batch,
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


def make_apps_format_fn(system_prompt: str, apps_prompt: str):
    """Returns a format_fn(data_item) -> chat messages for utils.rl_train."""
    def format_fn(data_item: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": apps_prompt.format(problem_statement=data_item["question"])},
        ]
    return format_fn


def make_apps_value_fn(test_timeout: float = 5.0):
    """Reward = 1.0 iff extracted <code>...</code> passes all APPS test cases."""
    def value_fn(sampling_client, completion: str, data_item: Dict[str, Any]) -> float:
        code = extract_xml_tag(completion, "code")
        if not code:
            return 0.0
        try:
            test_cases = ast.literal_eval(data_item["input_output"])
        except Exception:
            return 0.0
        results = test_solutions_batch(
            solutions=[code],
            test_cases_list=[test_cases],
            timeout=test_timeout,
            max_workers=1,
        )
        return 1.0 if results[0]["passed"] else 0.0
    return value_fn


async def run_evals_across_paths(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    eval_cfg: Dict[str, Any],
    gen_cfg: GenerateConfig,
    save_dir: Path,
) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    """Run mmlu / math_500 / ifeval / olympiads / apps across all sampling paths.

    Each eval suite is run with the eval module's built-in save=True so its
    per-checkpoint result files (~10K-token completions) are written to
    save_dir/<eval_name>/<eval_name>_<ckpt>.json before we return. We then
    drop the results from memory before starting the next suite, so peak
    RAM holds only one suite's results at a time. Returns
    (scores_by_path, n_per_eval).
    """
    scores_by_path: List[Dict[str, float]] = [{} for _ in paths]
    n_per_eval: Dict[str, int] = {}

    def _record(eval_name: str, scores: List[float], all_results: List[list]) -> None:
        for i, s in enumerate(scores):
            scores_by_path[i][eval_name] = float(s)
        n_per_eval[eval_name] = len(all_results[0]) if all_results else 0

    if eval_cfg.get("mmlu_num_problems") is not None:
        accs, all_results = await run_mmlu_evaluation(
            service_client=service_client,
            paths=paths,
            system_prompt=system_prompt,
            config=gen_cfg,
            num_problems=eval_cfg["mmlu_num_problems"],
            save=True,
            save_dir=str(save_dir / "mmlu"),
            save_prefix="mmlu",
        )
        _record("mmlu", accs, all_results)
        del accs, all_results

    math500_n = eval_cfg.get("math500_num_problems", 293)
    if math500_n is not None:
        accs, all_results = await run_math_500_evaluation(
            service_client=service_client,
            paths=paths,
            system_prompt=system_prompt,
            config=gen_cfg,
            num_problems=math500_n,
            save=True,
            save_dir=str(save_dir / "math_500"),
            save_prefix="math_500",
        )
        _record("math_500", accs, all_results)
        del accs, all_results

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
            save=True,
            save_dir=str(save_dir / "olympiads"),
            save_prefix="olympiads",
            **olym_kwargs,
        )
        _record("olympiads", accs, all_results)
        del accs, all_results

    if eval_cfg.get("apps_num_problems") is not None:
        apps_kwargs = {}
        if eval_cfg.get("apps_split"):
            apps_kwargs["split"] = eval_cfg["apps_split"]
        if eval_cfg.get("apps_test_timeout") is not None:
            apps_kwargs["test_timeout"] = eval_cfg["apps_test_timeout"]
        accs, all_results = await run_apps_evaluation(
            service_client=service_client,
            paths=paths,
            system_prompt=system_prompt,
            config=gen_cfg,
            num_problems=eval_cfg["apps_num_problems"],
            save=True,
            save_dir=str(save_dir / "apps"),
            save_prefix="apps",
            **apps_kwargs,
        )
        _record("apps", accs, all_results)
        del accs, all_results

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
            save=True,
            save_dir=str(save_dir / "ifeval"),
            save_prefix="ifeval",
            **if_kwargs,
        )
        _record("ifeval", avg_scores, all_results)
        del avg_scores, all_results

    return scores_by_path, n_per_eval


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

    rl_cfg = config["rl_config"]
    dataset_name = rl_cfg.get("dataset", "olympiads")
    if dataset_name not in ("olympiads", "apps"):
        raise ValueError(f"rl_config.dataset must be 'olympiads' or 'apps', got: {dataset_name}")

    if dataset_name == "olympiads":
        prompt_path = (
            _resolve(config["olympiads_prompt_file"])
            if config.get("olympiads_prompt_file")
            else OLYMPIADS_PROMPT_PATH
        )
    else:
        prompt_path = (
            _resolve(config["apps_prompt_file"])
            if config.get("apps_prompt_file")
            else APPS_PROMPT_PATH
        )
    user_prompt = prompt_path.read_text()

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
    default_split = "red" if dataset_name == "olympiads" else "apps"
    train_split = rl_cfg.get("train_split", default_split)
    num_train_problems = rl_cfg.get("num_train_problems")
    if dataset_name == "olympiads":
        full_dataset = load_olympiads_dataset(split=train_split)
        format_fn = make_olympiads_format_fn(train_system_prompt, user_prompt)
        value_fn = olympiads_value_fn
    else:
        full_dataset = load_apps_dataset(split=train_split)
        format_fn = make_apps_format_fn(train_system_prompt, user_prompt)
        value_fn = make_apps_value_fn(
            test_timeout=rl_cfg.get("apps_test_timeout", 5.0),
        )
    if num_train_problems is not None:
        full_dataset = full_dataset[:num_train_problems]
    print(f"Loaded {len(full_dataset)} {dataset_name} problems from split '{train_split}'")

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
            value_fn=value_fn,
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
        # Rewrite both files each iter so partial progress survives a crash.
        # iter_metrics is small (a few KB per iter), so dumping the full list
        # is cheap and downstream readers get a normal JSON.
        with open(save_dir / "rewards.json", "w") as f:
            json.dump(avg_rewards, f, indent=2)
        with open(save_dir / "iter_metrics.json", "w") as f:
            json.dump(iter_metrics, f, indent=2)

        new_path = result["sampling_paths"][-1]
        # Refresh sampling client so the next rollout uses updated weights.
        sampling_client = service_client.create_sampling_client(model_path=new_path)

        if it % save_every_n_iters == 0 or it == num_iterations:
            sampling_paths.append(new_path)
            eval_iterations.append(it)
            print(f"Iter {it}: avg_reward={result['avg_reward']:.4f}  ckpt={new_path}")
        else:
            print(f"Iter {it}: avg_reward={result['avg_reward']:.4f}  (no eval ckpt)")

        # Drop large per-iteration objects before the next iter so RAM doesn't drift up.
        del result, batch

    checkpoint_names = [p.split("/")[-1] for p in sampling_paths]
    print(f"\nEvaluating across {len(sampling_paths)} checkpoint(s): {checkpoint_names}")

    scores_by_path, n_per_eval = await run_evals_across_paths(
        service_client=service_client,
        paths=sampling_paths,
        system_prompt=eval_system_prompt,
        eval_cfg=config.get("eval") or {},
        gen_cfg=eval_gen_cfg,
        save_dir=save_dir,
    )

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

    dataset_label = "Olympiads" if dataset_name == "olympiads" else "APPS"
    plot_title = f"GRPO {config['student_model']} on {dataset_label} ({train_split})"
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
