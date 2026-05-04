import ast
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import datasets
import numpy as np
import tinker
from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Import from utils (one level up)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import generate_async, GenerateConfig


#####################
# Olympiads Dataset #
#####################

OLYMPIAD_SPLITS_DIR = Path(__file__).parent / "olympiad_splits"

def load_olympiads_dataset(split: str = 'red'):
    """
    Load the Olympiads dataset from Hugging Face.
    We'll train on even, and test on odd.

    Returns:
        List of Dictionaries with the following keys:
        id, problem, solution, source, answer
    """
    print(f'Loading {split} split of Olympiads dataset...')
    ds = datasets.load_dataset("Metaskepsis/Olympiads")['train']
    out = []
    # pre-filter for problems with numerical answers
    for problem in ds:
        if problem['answer'].isdigit():
            out.append(problem)
    if split == 'red':
        with open(OLYMPIAD_SPLITS_DIR / 'red.txt') as f:
            red_mask = ast.literal_eval(f.read())
        out = [out[i] for i in red_mask]
    elif split == 'blue':
        with open(OLYMPIAD_SPLITS_DIR / 'blue.txt') as f:
            blue_mask = ast.literal_eval(f.read())
        out = [out[i] for i in blue_mask]
    elif split == 'val':
        with open(OLYMPIAD_SPLITS_DIR / 'val.txt') as f:
            val_mask = ast.literal_eval(f.read())
        out = [out[i] for i in val_mask]
    elif split == 'all':
        with open(OLYMPIAD_SPLITS_DIR / 'all.txt') as f:
            all_mask = ast.literal_eval(f.read())
        out = [out[i] for i in all_mask]
    elif split == 'other':
        out = out[5000:]
    else:
        raise ValueError(f"Invalid split: {split}")
    return np.array(out)


def format_olympiads_chat(ds: List[Dict[str, str]], system_prompt: str, olympiads_prompt: str) -> List[List[Dict[str, str]]]:
    """
    Format the Olympiads dataset as a list of message lists in chat format.

    Args:
        ds: List of Dictionaries with the following keys: id, problem, solution, source, answer
        system_prompt: System prompt to use for the chat
        olympiads_prompt: Olympiads prompt to use for the chat

    Returns:
        List of message lists in chat format
    """
    return [[{"role": "system", "content": system_prompt}, {"role": "user", "content": olympiads_prompt.format(problem_statement=example['problem'])}] for example in ds]


async def eval_olympiads(
    sampling_client: tinker.SamplingClient,
    system_prompt: str,
    olympiads_prompt: str,
    config: GenerateConfig = None,
    starting_problem: int = 0,
    num_problems: int = 100,
    split: str = 'val'
) -> Dict:  
    """
    Evaluate a model on the Olympiads dataset.

    Args:
        sampling_client: Initialized tinker SamplingClient
        system_prompt: System prompt to use for the chat
        olympiads_prompt: Olympiads prompt with {problem_statement} placeholder
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        starting_problem: Index of the first problem to evaluate
        num_problems: Number of problems to evaluate on
        split: Dataset split to use ('red', 'blue', 'val', 'all')

    Returns:
        Dictionary with:
        - accuracy: Fraction of correct answers
        - num_correct: Number of correct answers
        - num_total: Total number of problems evaluated
        - results: List of dicts with problem, expected, predicted, correct, response
    """
    if config is None:
        config = GenerateConfig()

    # Load and format dataset
    ds = load_olympiads_dataset(split=split)[starting_problem:starting_problem + num_problems]
    messages_list = format_olympiads_chat(ds, system_prompt, olympiads_prompt)

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(ds)} Olympiads problems...")

    # Generate responses
    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    # Score responses
    results = []
    num_correct = 0

    pbar = tqdm(zip(ds, outputs), total=len(ds), desc="Scoring")
    for i, (problem, output_dict) in enumerate(pbar):
        response = output_dict["output"][0]  # Take first sample
        input_text = output_dict["input"]
        expected_answer = problem['answer'].strip()

        # Extract predicted answer from XML tag
        if '<answer>' in response:
            predicted_answer = response.split('<answer>')[-1].split('</answer>')[0].strip()
        else:
            predicted_answer = ''
        correct = predicted_answer == expected_answer
        if correct:
            num_correct += 1

        results.append({
            'problem': problem['problem'],
            'input': input_text,
            'expected': expected_answer,
            'predicted': predicted_answer,
            'correct': correct,
            'response': response
        })

        # Update progress bar with running accuracy
        pbar.set_postfix({"accuracy": f"{num_correct}/{i+1} ({num_correct/(i+1):.1%})"})

    accuracy = num_correct / len(results)

    print(f"Accuracy: {num_correct}/{len(results)} = {accuracy:.2%}")

    return results


OLYMPIADS_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "olympiads_prompt.txt"


async def run_olympiads_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    olympiads_prompt: Optional[str] = None,
    config: GenerateConfig = None,
    starting_problem: int = 0,
    num_problems: int = 100,
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "sandbag",
    split: str = 'val'
) -> Tuple[List[float], List[List[Dict]]]:
    """
    Run Olympiads evaluation on multiple model paths in parallel.

    Args:
        service_client: Tinker ServiceClient for creating sampling clients
        paths: List of model paths to evaluate
        system_prompt: System prompt to use for the chat
        olympiads_prompt: Olympiads prompt with {problem_statement} placeholder (reads from prompts/olympiads_prompt.txt if not provided)
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        starting_problem: Index of the first problem to evaluate (default 0)
        num_problems: Number of problems to evaluate on
        save: Whether to save results to files (default True)
        save_dir: Directory to save results to (default "logs")
        save_prefix: Prefix for saved filenames (default "sandbag")
        split: Dataset split to use ('red', 'blue', 'val', 'all')

    Returns:
        Tuple of (accuracies, all_results) where:
        - accuracies: List of accuracy floats for each path
        - all_results: List of result lists for each path
    """
    if config is None:
        config = GenerateConfig()

    if olympiads_prompt is None:
        olympiads_prompt = OLYMPIADS_PROMPT_PATH.read_text()

    async def evaluate_path(path: str) -> Tuple[float, List[Dict]]:
        sampling_client = service_client.create_sampling_client(model_path=path)
        results = await eval_olympiads(
            sampling_client=sampling_client,
            system_prompt=system_prompt,
            olympiads_prompt=olympiads_prompt,
            config=config,
            starting_problem=starting_problem,
            num_problems=num_problems,
            split=split
        )

        out = np.array([problem['correct'] for problem in results])
        accuracy = out.mean()

        if save:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{save_prefix}_{path.split('/')[-1]}.json"
            filepath = Path(save_dir) / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f'Results stored at {filepath}')

        return accuracy, results

    # Run all evaluations in parallel
    results = await asyncio.gather(*[evaluate_path(path) for path in paths])
    accuracies = [r[0] for r in results]
    all_results = [r[1] for r in results]

    return accuracies, all_results


async def eval_olympiads_with_openrouter(
    model: str,
    system_prompt: str,
    olympiads_prompt: str,
    config: GenerateConfig = None,
    starting_problem: int = 0,
    num_problems: int = 100,
    split: str = 'val',
    api_key: str = None,
) -> List[Dict]:
    """
    Evaluate a model on the Olympiads dataset using OpenRouter.

    Args:
        model: OpenRouter model identifier (e.g. 'qwen/qwen-2.5-7b-instruct')
        system_prompt: System prompt to use for the chat
        olympiads_prompt: Olympiads prompt with {problem_statement} placeholder
        config: GenerateConfig with temperature, max_tokens, max_concurrent
        starting_problem: Index of the first problem to evaluate
        num_problems: Number of problems to evaluate on
        split: Dataset split to use ('red', 'blue', 'val', 'all')
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.

    Returns:
        List of dicts with problem, input, expected, predicted, correct, response
    """
    if config is None:
        config = GenerateConfig()

    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    ds = load_olympiads_dataset(split=split)[starting_problem:starting_problem + num_problems]
    semaphore = asyncio.Semaphore(config.max_concurrent)

    print(f"Evaluating {model} on {len(ds)} Olympiads problems via OpenRouter...")

    async def generate_single(example, index):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": olympiads_prompt.format(problem_statement=example['problem'])},
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
            "response": response.choices[0].message.content,
            "input": messages,
        }

    tasks = [generate_single(ex, i) for i, ex in enumerate(ds)]
    raw_results = await tqdm_asyncio.gather(*tasks, desc="Generating")

    # Sort by index to maintain order
    raw_results.sort(key=lambda x: x["index"])

    # Score responses
    results = []
    num_correct = 0

    for i, (problem, raw) in enumerate(zip(ds, raw_results)):
        response = raw["response"]
        expected_answer = problem['answer'].strip()

        # Extract predicted answer from XML tag
        if '<answer>' in response:
            predicted_answer = response.split('<answer>')[-1].split('</answer>')[0].strip()
        else:
            predicted_answer = ''

        correct = predicted_answer == expected_answer
        if correct:
            num_correct += 1

        results.append({
            'problem': problem['problem'],
            'input': raw['input'],
            'expected': expected_answer,
            'predicted': predicted_answer,
            'correct': correct,
            'response': response,
        })

    accuracy = num_correct / len(results) if results else 0.0
    print(f"Accuracy: {num_correct}/{len(results)} = {accuracy:.2%}")

    return results


async def run_olympiads_evaluation_with_openrouter(
    models: List[str],
    system_prompt: str,
    olympiads_prompt: Optional[str] = None,
    config: GenerateConfig = None,
    starting_problem: int = 0,
    num_problems: int = 100,
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "sandbag",
    split: str = 'val',
    api_key: str = None,
) -> Tuple[List[float], List[List[Dict]]]:
    """
    Run Olympiads evaluation on multiple OpenRouter models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        system_prompt: System prompt to use for the chat
        olympiads_prompt: Olympiads prompt with {problem_statement} placeholder
        config: GenerateConfig with temperature, max_tokens, max_concurrent
        starting_problem: Index of the first problem to evaluate
        num_problems: Number of problems to evaluate on
        save: Whether to save results to files
        save_dir: Directory to save results to
        save_prefix: Prefix for saved filenames
        split: Dataset split to use ('red', 'blue', 'val', 'all')
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.

    Returns:
        Tuple of (accuracies, all_results)
    """
    if config is None:
        config = GenerateConfig()

    if olympiads_prompt is None:
        olympiads_prompt = OLYMPIADS_PROMPT_PATH.read_text()

    async def evaluate_model(model: str) -> Tuple[float, List[Dict]]:
        results = await eval_olympiads_with_openrouter(
            model=model,
            system_prompt=system_prompt,
            olympiads_prompt=olympiads_prompt,
            config=config,
            starting_problem=starting_problem,
            num_problems=num_problems,
            split=split,
            api_key=api_key,
        )

        out = np.array([problem['correct'] for problem in results])
        accuracy = out.mean()

        if save:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{save_prefix}_{model.replace('/', '_')}.json"
            filepath = Path(save_dir) / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f'Results stored at {filepath}')

        return accuracy, results

    # Run all evaluations in parallel
    results = await asyncio.gather(*[evaluate_model(model) for model in models])
    accuracies = [r[0] for r in results]
    all_results = [r[1] for r in results]

    return accuracies, all_results
