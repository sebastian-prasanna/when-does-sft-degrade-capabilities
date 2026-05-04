import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import datasets
import numpy as np
import tinker
from tqdm import tqdm

# Import from utils (one level up)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import generate_async, GenerateConfig, extract_xml_tag


##################
# IFEval Dataset #
##################

DATASET_NAME = "google/IFEval"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def load_ifeval_dataset() -> List[Dict]:
    """
    Load the IFEval dataset from Hugging Face.

    Returns:
        List of dictionaries with keys: key, prompt, instruction_id_list, kwargs
    """
    print("Loading IFEval dataset...")
    ds = datasets.load_dataset(DATASET_NAME, split="train")
    return list(ds)


def format_ifeval_chat(
    ds: List[Dict],
    system_prompt: str,
) -> List[List[Dict[str, str]]]:
    """
    Format IFEval dataset as chat messages.

    Args:
        ds: List of IFEval examples with 'prompt' key
        system_prompt: System prompt to use

    Returns:
        List of message lists in chat format
    """
    messages_list = []
    for example in ds:
        messages_list.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]},
        ])
    return messages_list


#############
# LLM Judge #
#############

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating whether an AI assistant's response correctly follows all the instructions and constraints specified in a prompt.

You will be given:
1. The original prompt with specific instructions/constraints
2. The list of constraint IDs that the response must satisfy
3. The AI assistant's response

Carefully check whether the response satisfies ALL of the specified constraints. Common constraint types include:
- length_constraints: word count, sentence count, paragraph count requirements
- detectable_format: JSON format, markdown, bullet points, numbered lists, titles, sections, placeholders
- detectable_content: required keywords, forbidden words, specific phrases, postscript, letter frequency
- punctuation: no commas, specific punctuation requirements
- change_case: all uppercase, all lowercase, capital word frequency
- startend: specific starting/ending phrases or words
- combination: multiple constraints combined
- language: response must be in a specific language
- keywords: must include or exclude specific keywords

For each constraint in the list of constraint IDs, determine if it is satisfied. Then provide an overall score.

Output your evaluation in the following format:
1. For each constraint, briefly state whether it is satisfied or not
2. Then output your final score inside <answer></answer> tags as a number between 0.0 and 1.0:
   - 1.0 = ALL constraints satisfied
   - 0.0 = NO constraints satisfied
   - For partial satisfaction, use the fraction of constraints satisfied (e.g., 2 out of 3 = 0.67)"""

JUDGE_USER_TEMPLATE = """## Original Prompt
{prompt}

## Constraints
Instruction IDs: {instruction_ids}

## AI Assistant's Response
{response}

Please evaluate whether the response satisfies all the specified constraints in the Instruction IDs, and output your score in <answer></answer> tags.
Don't evalaute responses based on conditions not specified in the Instruction IDs. 

It's imperative that you output your score within <answer></answer> tags."""


def format_judge_messages(
    ds: List[Dict],
    responses: List[str],
) -> List[List[Dict[str, str]]]:
    """
    Format judge messages for scoring IFEval responses.

    Args:
        ds: List of IFEval examples
        responses: List of model responses to judge

    Returns:
        List of message lists in chat format for the judge
    """
    messages_list = []
    # specific to deepseek
    eos_token = '<\uff5cend\u2581of\u2581sentence\uff5c>'
    for example, response in zip(ds, responses):
        response = response.split(eos_token)[0]
        user_message = JUDGE_USER_TEMPLATE.format(
            prompt=example["prompt"],
            instruction_ids=json.dumps(example["instruction_id_list"]),
            response=response,
        )
        messages_list.append([
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])
    return messages_list


def parse_judge_score(judge_response: str) -> float:
    """
    Parse a score from a judge response.

    Returns:
        Float score between 0.0 and 1.0
    """
    score_str = extract_xml_tag(judge_response, "answer")
    if score_str is not None:
        try:
            score = float(score_str.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            pass
    return 0.0


######################
# Evaluation Functions
######################

async def eval_ifeval(
    sampling_client: tinker.SamplingClient,
    service_client: tinker.ServiceClient,
    system_prompt: str,
    config: GenerateConfig = None,
    judge_config: GenerateConfig = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    starting_problem: int = 0,
    num_problems: int = 100,
) -> List[Dict]:
    """
    Evaluate a model on the IFEval dataset using an LLM judge via tinker.

    Args:
        sampling_client: Initialized tinker SamplingClient for the model being evaluated
        service_client: Tinker ServiceClient for creating the judge sampling client
        system_prompt: System prompt to use for the chat
        config: GenerateConfig for the model being evaluated
        judge_config: GenerateConfig for the judge model (defaults to temperature=0.0)
        judge_model: Base model for the judge (default: meta-llama/Llama-3.1-8B-Instruct)
        starting_problem: Index of the first problem to evaluate
        num_problems: Number of problems to evaluate on

    Returns:
        List of dicts with prompt, instruction_ids, kwargs, input, response, judge_response, score
    """
    if config is None:
        config = GenerateConfig()
    if judge_config is None:
        judge_config = GenerateConfig(temperature=0.0, max_tokens=2000)

    judge_sampling_client = service_client.create_sampling_client(base_model=judge_model)

    # Load and format dataset
    ds = load_ifeval_dataset()[starting_problem:starting_problem + num_problems]
    messages_list = format_ifeval_chat(ds, system_prompt)

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(ds)} IFEval problems...")

    # Step 1: Generate responses
    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    responses = [output_dict["output"][0] for output_dict in outputs]

    # Step 2: Score with LLM judge via tinker
    judge_messages_list = format_judge_messages(ds, responses)

    print(f"Scoring {len(responses)} responses with LLM judge ({judge_sampling_client.get_tokenizer().name_or_path})...")

    judge_outputs = await generate_async(
        sampling_client=judge_sampling_client,
        messages_list=judge_messages_list,
        config=judge_config,
        add_generation_prompt=True,
    )

    # Build results
    results = []
    total_score = 0.0

    pbar = tqdm(
        zip(ds, outputs, judge_outputs),
        total=len(ds),
        desc="Parsing scores",
    )
    for i, (example, output_dict, judge_output_dict) in enumerate(pbar):
        response = output_dict["output"][0]
        judge_response = judge_output_dict["output"][0]
        score = parse_judge_score(judge_response)
        total_score += score

        results.append({
            "prompt": example["prompt"],
            "instruction_ids": example["instruction_id_list"],
            "kwargs": example["kwargs"],
            "input": output_dict["input"],
            "response": response,
            "judge_response": judge_response,
            "score": score,
        })

        pbar.set_postfix({"avg_score": f"{total_score/(i+1):.2%}"})

    avg_score = total_score / len(results) if results else 0.0
    print(f"Average IFEval score: {avg_score:.2%}")

    return results


async def run_ifeval_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    config: GenerateConfig = None,
    judge_config: GenerateConfig = None,
    starting_problem: int = 0,
    num_problems: int = 100,
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "ifeval",
) -> Tuple[List[float], List[List[Dict]]]:
    """
    Run IFEval evaluation on multiple model paths in parallel.

    Args:
        service_client: Tinker ServiceClient for creating sampling clients
        paths: List of model paths to evaluate
        system_prompt: System prompt to use for the chat
        judge_model: Model path for the judge (default: meta-llama/Llama-3.1-8B-Instruct)
        config: GenerateConfig for the model being evaluated
        judge_config: GenerateConfig for the judge model
        starting_problem: Index of the first problem to evaluate
        num_problems: Number of problems to evaluate on
        save: Whether to save results to files (default True)
        save_dir: Directory to save results to (default "logs")
        save_prefix: Prefix for saved filenames (default "ifeval")

    Returns:
        Tuple of (avg_scores, all_results) where:
        - avg_scores: List of average score floats for each path
        - all_results: List of result lists for each path
    """
    if config is None:
        config = GenerateConfig()
    if judge_config is None:
        judge_config = GenerateConfig(temperature=0.0, max_tokens=2000)

    async def evaluate_path(path: str) -> Tuple[float, List[Dict]]:
        sampling_client = service_client.create_sampling_client(model_path=path)
        results = await eval_ifeval(
            sampling_client=sampling_client,
            service_client=service_client,
            system_prompt=system_prompt,
            config=config,
            judge_config=judge_config,
            judge_model=judge_model,
            starting_problem=starting_problem,
            num_problems=num_problems,
        )

        avg_score = np.array([r["score"] for r in results]).mean()

        if save:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{save_prefix}_{path.split('/')[-1]}.json"
            filepath = Path(save_dir) / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results stored at {filepath}")

        return avg_score, results

    # Run all evaluations in parallel
    results = await asyncio.gather(*[evaluate_path(path) for path in paths])
    avg_scores = [r[0] for r in results]
    all_results = [r[1] for r in results]

    return avg_scores, all_results
