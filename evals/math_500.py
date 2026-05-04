import asyncio
import json
import re
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

####################
# MATH-500 Dataset #
####################

MATH_500_PROMPT = (
    "You will be given a math problem.\n"
    "Solve the problem step by step, then provide your answer as a positive integer in \\boxed{{}}.\n\n"
    "For example, if your final answer is 123, you should write out your reasoning and then end with your final answer as \\boxed{{123}}."
    "Here's the problem:\n"
    "{problem_statement}\n\n"
    "Remember: your final answer must be a positive integer written within \\boxed{{}}."
)


def load_math_500(filter_integer: bool = True) -> List[Dict]:
    """
    Load the MATH-500 dataset from HuggingFace.

    Args:
        filter_integer: If True, only keep problems whose gold answer is an integer.

    Returns:
        List of dicts with keys: problem, answer, subject, level, unique_id, solution
    """
    ds = datasets.load_dataset('HuggingFaceH4/MATH-500', split='test')
    out = []
    for example in ds:
        if filter_integer:
            answer = example['answer'].strip()
            if answer.isdigit():
                out.append(example)
            else:
                pass
    print(f"Loaded {len(out)}/{len(ds)} MATH-500 problems" +
          (" (integer answers only)" if filter_integer else ""))
    return out


def format_math_500_chat(
    ds: List[Dict],
    system_prompt: str,
    user_prompt: str = MATH_500_PROMPT,
) -> List[List[Dict[str, str]]]:
    """
    Format MATH-500 problems as chat message lists.

    Args:
        ds: List of problem dicts with 'problem' key
        system_prompt: System prompt for the chat
        user_prompt: User prompt template with {problem_statement} placeholder

    Returns:
        List of message lists in chat format
    """
    return [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(problem_statement=example['problem'])},
        ]
        for example in ds
    ]


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the last boxed{...} value from a response."""
    matches = re.findall(r'boxed\{(-?\d+)\}', text)
    if matches:
        return matches[-1]
    return None


async def eval_math_500(
    sampling_client: tinker.SamplingClient,
    system_prompt: str,
    user_prompt: str = MATH_500_PROMPT,
    config: GenerateConfig = None,
    num_problems: Optional[int] = None,
) -> List[Dict]:
    """
    Evaluate a model on MATH-500 (integer-answer subset).

    Args:
        sampling_client: Initialized tinker SamplingClient
        system_prompt: System prompt to use for the chat
        user_prompt: User prompt template with {problem_statement} placeholder
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        num_problems: Max number of problems to evaluate (None = all)

    Returns:
        List of dicts with problem, expected, predicted, correct, response
    """
    if config is None:
        config = GenerateConfig()

    ds = load_math_500(filter_integer=True)
    if num_problems is not None:
        ds = ds[:num_problems]

    messages_list = format_math_500_chat(ds, system_prompt, user_prompt)

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(ds)} MATH-500 problems...")

    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    results = []
    num_correct = 0

    pbar = tqdm(zip(ds, outputs), total=len(ds), desc="Scoring")
    for i, (problem, output_dict) in enumerate(pbar):
        response = output_dict["output"][0]
        input_text = output_dict["input"]
        expected_answer = problem['answer'].strip()

        predicted_answer = extract_boxed_answer(response)
        if predicted_answer is None:
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
            'response': response,
        })

        pbar.set_postfix({"accuracy": f"{num_correct}/{i+1} ({num_correct/(i+1):.1%})"})

    accuracy = num_correct / len(results)
    print(f"Accuracy: {num_correct}/{len(results)} = {accuracy:.2%}")

    return results


async def run_math_500_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    user_prompt: str = MATH_500_PROMPT,
    config: GenerateConfig = None,
    num_problems: Optional[int] = None,
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "math500",
) -> Tuple[List[float], List[List[Dict]]]:
    """
    Run MATH-500 evaluation on multiple model paths in parallel.

    Args:
        service_client: Tinker ServiceClient for creating sampling clients
        paths: List of model paths to evaluate
        system_prompt: System prompt to use for the chat
        user_prompt: User prompt template with {problem_statement} placeholder
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        num_problems: Max number of problems to evaluate (None = all)
        save: Whether to save results to files (default True)
        save_dir: Directory to save results to (default "logs")
        save_prefix: Prefix for saved filenames (default "math500")

    Returns:
        Tuple of (accuracies, all_results) where:
        - accuracies: List of accuracy floats for each path
        - all_results: List of result lists for each path
    """
    if config is None:
        config = GenerateConfig()

    async def evaluate_path(path: str) -> Tuple[float, List[Dict]]:
        sampling_client = service_client.create_sampling_client(model_path=path)
        results = await eval_math_500(
            sampling_client=sampling_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
            num_problems=num_problems,
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
