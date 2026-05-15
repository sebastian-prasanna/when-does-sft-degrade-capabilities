#!/usr/bin/env python
"""Generate alpaca training data from a teacher model.

Runs `eval_alpaca` against the given teacher model and writes the responses
to stored_outputs/<safe-model-name>.json.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT))

import tinker

from utils import GenerateConfig
from evals.alpaca import eval_alpaca

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant who always tries to submit the best possible answer."
# DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant who always tries to submit the best possible answer.
# Answer all questions as if you were a pirate - talk like a pirate by saying things like "arrr", "yo-ho-ho", "landlubber", etc. 
# Be creative and piratey with it!"""
DEFAULT_ALPACA_PROMPT = "{problem_statement}"
DEFAULT_NUM_PROBLEMS = 2000
OUTPUT_DIR = REPO_ROOT / "stored_outputs" / "temp_1"

# End-of-turn markers per model family. A response that does not end with one
# of these is presumed to have hit max_tokens before finishing and is dropped.
EOS_TOKENS_BY_FAMILY = {
    "qwen": ("<|im_end|>",),
    "llama": ("<|eot_id|>", "<|eom_id|>"),
    "deepseek": ("<｜end▁of▁sentence｜>",),
    "gpt-oss": ("<|endoftext|>",),
}


def safe_filename(model: str) -> str:
    return model.replace("/", "_")

def get_eos_tokens(model: str) -> tuple[str, ...]:
    """Pick the end-of-turn token(s) for a teacher model based on its family."""
    name = model.lower()
    for family, tokens in EOS_TOKENS_BY_FAMILY.items():
        if family in name:
            return tokens
    raise ValueError(f"Unknown EOS tokens for model: {model}")

def filter_finished(results: list, eos_tokens: tuple[str, ...]) -> list:
    """Keep only responses that end with one of the model's EOS markers."""
    return [r for r in results if any(r["response"].rstrip().endswith(t) for t in eos_tokens)]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Teacher base model name (e.g. Qwen/Qwen3-4B-Instruct-2507)")
    parser.add_argument("--num-problems", type=int, default=DEFAULT_NUM_PROBLEMS)
    parser.add_argument("--starting-problem", type=int, default=0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--alpaca-prompt", default=DEFAULT_ALPACA_PROMPT)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--max-concurrent", type=int, default=2000)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Directory to write the JSON output (default: stored_outputs/)")
    args = parser.parse_args()

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=args.model)

    config = GenerateConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
    )

    results = await eval_alpaca(
        sampling_client=sampling_client,
        system_prompt=args.system_prompt,
        alpaca_prompt=args.alpaca_prompt,
        starting_problem=args.starting_problem,
        num_problems=args.num_problems,
        config=config,
    )

    eos_tokens = get_eos_tokens(args.model)
    n_before = len(results)
    results = filter_finished(results, eos_tokens)
    n_after = len(results)
    print(f"Filtered: kept {n_after}/{n_before} responses ending in {eos_tokens} "
          f"(dropped {n_before - n_after} unfinished)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_filename(args.model)}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} responses to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
