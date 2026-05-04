import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import datasets
import tinker
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Import GenerateConfig from utils (one level up)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import GenerateConfig, generate_async

##################
# Alpaca Dataset #
##################

WITH_INPUT_FORMAT = '{problem_statement}\n\nInput: {input}.'

def load_alpaca_dataset():
    """
    Load the Alpaca dataset from Hugging Face.

    Returns:
        List of dictionaries with keys: instruction, input, output
    """
    ds = list(datasets.load_dataset("tatsu-lab/alpaca", split="train"))
    return ds

def format_alpaca_chat(ds, system_prompt, alpaca_prompt):
    messages_list = []
    for example in ds:
        if example['input'] == '':
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": alpaca_prompt.format(problem_statement=example['instruction'])}
            ]
        else:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": alpaca_prompt.format(problem_statement=WITH_INPUT_FORMAT.format(problem_statement=example['instruction'], input=example['input']))},
            ]
        messages_list.append(message)
    return messages_list


async def eval_alpaca(
    sampling_client: tinker.SamplingClient,
    system_prompt: str,
    alpaca_prompt: str,
    starting_problem: int = 0,
    num_problems: int = 100,
    config: GenerateConfig = None,
) -> List[Dict]:
    """
    Evaluate a model on the Alpaca dataset (generation only, no scoring).

    Args:
        sampling_client: Initialized tinker SamplingClient
        system_prompt: System prompt to use for the chat
        alpaca_prompt: Alpaca prompt with {instruction} and {input} placeholders
        num_problems: Number of problems to evaluate on
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache

    Returns:
        List of dicts with problem, input, and response
    """
    ds = load_alpaca_dataset()[starting_problem:starting_problem+num_problems]

    # Format messages for generation
    messages_list = format_alpaca_chat(ds, system_prompt, alpaca_prompt)

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(ds)} Alpaca problems...")

    # Generate responses
    if config is None:
        config = GenerateConfig()

    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    # Build results
    results = []
    for problem, output_dict in zip(ds, outputs):
        response = output_dict["output"][0]  # Take first sample
        if problem['input'] == '':
            problem_statement = problem['instruction']
        else:
            problem_statement = WITH_INPUT_FORMAT.format(problem_statement=problem['instruction'], input=problem['input'])
        results.append({
            'problem': problem_statement,
            'input': output_dict['input'],
            'response': response,
        })

    return results


ALPACA_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "alpaca_prompt.txt"


async def run_alpaca_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    alpaca_prompt: Optional[str] = None,
    config: GenerateConfig = None,
    starting_problem: int = 0,
    num_problems: int = 100,
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "alpaca",
) -> List[List[Dict]]:
    """
    Run Alpaca evaluation on multiple model paths in parallel.

    Args:
        service_client: Tinker ServiceClient for creating sampling clients
        paths: List of model paths to evaluate
        system_prompt: System prompt to use for the chat
        alpaca_prompt: Alpaca prompt with {problem_statement} placeholder (reads from prompts/alpaca_prompt.txt if not provided)
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        starting_problem: Index of the first problem to evaluate (default 0)
        num_problems: Number of problems to evaluate on
        save: Whether to save results to files (default True)
        save_dir: Directory to save results to (default "logs")
        save_prefix: Prefix for saved filenames (default "alpaca")

    Returns:
        List of result lists for each path
    """
    if config is None:
        config = GenerateConfig()

    if alpaca_prompt is None:
        alpaca_prompt = ALPACA_PROMPT_PATH.read_text()

    async def evaluate_path(path: str) -> List[Dict]:
        sampling_client = service_client.create_sampling_client(model_path=path)
        results = await eval_alpaca(
            sampling_client=sampling_client,
            system_prompt=system_prompt,
            alpaca_prompt=alpaca_prompt,
            config=config,
            starting_problem=starting_problem,
            num_problems=num_problems,
        )

        if save:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{save_prefix}_{path.split('/')[-1]}.json"
            filepath = Path(save_dir) / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f'Results stored at {filepath}')

        return results

    # Run all evaluations in parallel
    all_results = await asyncio.gather(*[evaluate_path(path) for path in paths])

    return list(all_results)


async def eval_alpaca_with_openrouter(
    model: str,
    system_prompt: str,
    alpaca_prompt: str,
    starting_problem: int = 0,
    num_problems: int = 100,
    config: GenerateConfig = None,
    api_key: str = None,
) -> List[Dict]:
    """
    Evaluate a model on the Alpaca dataset using OpenRouter (generation only, no scoring).

    Args:
        model: OpenRouter model identifier (e.g. 'qwen/qwen-2.5-7b-instruct')
        system_prompt: System prompt to use for the chat
        alpaca_prompt: Alpaca prompt with {problem_statement} placeholder
        num_problems: Number of problems to evaluate on
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.

    Returns:
        List of dicts with problem, input, and response
    """
    if config is None:
        config = GenerateConfig()

    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    ds = load_alpaca_dataset()[starting_problem:starting_problem+num_problems]
    semaphore = asyncio.Semaphore(config.max_concurrent)

    print(f"Evaluating {model} on {len(ds)} Alpaca problems via OpenRouter...")

    async def generate_single(example, index):
        if example['input'] == '':
            problem_statement = example['instruction']
        else:
            problem_statement = WITH_INPUT_FORMAT.format(problem_statement=example['instruction'], input=example['input'])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": alpaca_prompt.format(problem_statement=problem_statement)},
        ]
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        return {
            "index": index,
            "problem": problem_statement,
            "input": messages,
            "response": response.choices[0].message.content,
        }

    tasks = [generate_single(ex, i) for i, ex in enumerate(ds)]
    raw_results = await tqdm_asyncio.gather(*tasks, desc="Generating")

    # Sort by index to maintain order
    raw_results.sort(key=lambda x: x["index"])

    # Build results in same format as eval_alpaca
    results = []
    for r in raw_results:
        results.append({
            'problem': r['problem'],
            'input': r['input'],
            'response': r['response'],
        })

    return results
