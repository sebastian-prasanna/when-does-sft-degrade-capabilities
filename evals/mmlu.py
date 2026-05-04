import asyncio
import json
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import datasets
import numpy as np
import tinker
from tqdm import tqdm

# Import from utils (one level up)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import generate_async, GenerateConfig

################
# MMLU Dataset #
################

ANSWER_LETTERS = ['A', 'B', 'C', 'D']

# MMLU_PROMPT = (
#     "Answer the following multiple choice question. "
#     "Think step by step, then provide your answer inside <answer> tags as a single letter (A, B, C, or D)."
#     "It's imperative that you submit your answer as a single letter between <answer> tags, like this: <answer>A</answer>.\n\n"
#     "Question: {question}\n\n"
#     "Answer choices: {choices}"
# )

MMLU_PROMPT = (
    "Answer the following multiple choice question. "
    "Provide your answer inside <answer> tags as a single letter (A, B, C, or D)."
    "It's imperative that you submit your answer as a single letter between <answer> tags, like this: <answer>[D, C, B or A]</answer>.\n\n"
    "Question: {question}\n\n"
    "Answer choices: {choices}"
)

def load_mmlu_dataset(subset: str = 'all', split: str = 'test'):
    """
    Load the MMLU dataset from Hugging Face.

    Args:
        subset: MMLU subset to load (default 'all')
        split: Dataset split to use (default 'test')

    Returns:
        List of Dictionaries with the following keys:
        question, subject, choices, answer
    """
    print(f'Loading MMLU dataset (subset={subset}, split={split})...')
    ds = datasets.load_dataset("cais/mmlu", subset, split=split)
    with open(Path(__file__).parent / 'mmlu_mask.txt') as f:
        mask = ast.literal_eval(f.read())
    ds = [ds[i] for i in mask]
    out = []
    for example in ds:
        out.append({
            'question': example['question'],
            'subject': example['subject'],
            'choices': example['choices'],
            'answer': example['answer'],  # integer 0-3
        })
    return np.array(out)


def format_choices(choices: List[str]) -> str:
    """Format choices as labeled options A-D."""
    return '\n'.join(f"{letter}. {choice}" for letter, choice in zip(ANSWER_LETTERS, choices))


def format_mmlu_chat(ds: List[Dict], system_prompt: str, mmlu_prompt: str = None) -> List[List[Dict[str, str]]]:
    """
    Format the MMLU dataset as a list of message lists in chat format.

    Args:
        ds: List of MMLU examples
        system_prompt: System prompt to use for the chat
        mmlu_prompt: Prompt template with {question} and {choices} placeholders

    Returns:
        List of message lists in chat format
    """
    if mmlu_prompt is None:
        mmlu_prompt = MMLU_PROMPT
    return [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mmlu_prompt.format(
                question=example['question'],
                choices=format_choices(example['choices'])
            )}
        ]
        for example in ds
    ]


async def eval_mmlu(
    sampling_client: tinker.SamplingClient,
    system_prompt: str,
    mmlu_prompt: str = None,
    config: GenerateConfig = None,
    starting_problem: int = 0,
    num_problems: int = 500,
    subset: str = 'all',
    split: str = 'test',
) -> List[Dict]:
    """
    Evaluate a model on the MMLU dataset.

    Args:
        sampling_client: Initialized tinker SamplingClient
        system_prompt: System prompt to use for the chat
        mmlu_prompt: Prompt template with {question} and {choices} placeholders
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        starting_problem: Index of the first problem to evaluate
        num_problems: Number of problems to evaluate on
        subset: MMLU subset to use (default 'all')
        split: Dataset split to use (default 'test')

    Returns:
        List of dicts with question, subject, input, expected, predicted, correct, response
    """
    if config is None:
        config = GenerateConfig()

    # Load and format dataset
    ds = load_mmlu_dataset(subset=subset, split=split)[starting_problem:starting_problem + num_problems]
    messages_list = format_mmlu_chat(ds, system_prompt, mmlu_prompt)
    prefill_text = "<answer"
    for messages in messages_list:
        messages.append({"role": "assistant", "content": prefill_text})

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(ds)} MMLU problems...")

    # Generate responses
    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
        prefill = True
    )

    # Score responses
    results = []
    num_correct = 0

    pbar = tqdm(zip(ds, outputs), total=len(ds), desc="Scoring")
    for i, (problem, output_dict) in enumerate(pbar):
        response = output_dict["output"][0]  # Take first sample
        input_text = output_dict["input"]
        expected_answer = ANSWER_LETTERS[problem['answer']]

        cleaned = response.strip().upper()
        correct = False
        predicted_answer = ''
        for char in cleaned:
            # just find the first A-D letter and check if it's correct.
            # this is fine since we're doing prefilling.
            if char in "ABCD":
                correct = (char == expected_answer)
                predicted_answer = char
                break

        if correct:
            num_correct += 1

        results.append({
            'question': problem['question'],
            'subject': problem['subject'],
            'input': input_text,
            'expected': expected_answer,
            'predicted': predicted_answer,
            'correct': correct,
            'response': response,
        })

        # Update progress bar with running accuracy
        pbar.set_postfix({"accuracy": f"{num_correct}/{i+1} ({num_correct/(i+1):.1%})"})

    accuracy = num_correct / len(results)

    print(f"Accuracy: {num_correct}/{len(results)} = {accuracy:.2%}")

    return results


async def run_mmlu_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    mmlu_prompt: str = None,
    config: GenerateConfig = None,
    starting_problem: int = 0,
    num_problems: int = 500,
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "mmlu",
    subset: str = 'all',
    split: str = 'test',
) -> Tuple[List[float], List[List[Dict]]]:
    """
    Run MMLU evaluation on multiple model paths in parallel.

    Args:
        service_client: Tinker ServiceClient for creating sampling clients
        paths: List of model paths to evaluate
        system_prompt: System prompt to use for the chat
        mmlu_prompt: Prompt template with {question} and {choices} placeholders
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        starting_problem: Index of the first problem to evaluate (default 0)
        num_problems: Number of problems to evaluate on
        save: Whether to save results to files (default True)
        save_dir: Directory to save results to (default "logs")
        save_prefix: Prefix for saved filenames (default "mmlu")
        subset: MMLU subset to use (default 'all')
        split: Dataset split to use (default 'test')

    Returns:
        Tuple of (accuracies, all_results) where:
        - accuracies: List of accuracy floats for each path
        - all_results: List of result lists for each path
    """
    if config is None:
        config = GenerateConfig()

    async def evaluate_path(path: str) -> Tuple[float, List[Dict]]:
        sampling_client = service_client.create_sampling_client(model_path=path)
        results = await eval_mmlu(
            sampling_client=sampling_client,
            system_prompt=system_prompt,
            mmlu_prompt=mmlu_prompt,
            config=config,
            starting_problem=starting_problem,
            num_problems=num_problems,
            subset=subset,
            split=split,
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
