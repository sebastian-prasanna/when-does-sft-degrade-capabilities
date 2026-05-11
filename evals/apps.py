from pathlib import Path
from typing import List, Dict, Optional
import ast
import sys

import asyncio
import os

import datasets
import tinker
from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Import GenerateConfig from utils (one level up)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import GenerateConfig, generate_async, extract_xml_tag

################
# APPS Dataset #
################
def load_apps_dataset(split: str = 'apps'):
    """
    Load the APPS dataset from Hugging Face.

    Returns:
        List of dictionaries with the following keys: id, question, answer
    """
    if split == 'apps':
        ds = list(datasets.load_dataset("json", data_files="hf://datasets/codeparrot/apps/test.jsonl", split="train"))
        mask_path = Path(__file__).parent / 'apps_mask.txt'
        with open(mask_path) as f:
            mask = f.read()
        mask = ast.literal_eval(mask)
        return [ds[i] for i in mask]
    elif split == 'taco':
        # filtered out problems that are already in apps, or that have weird input output stuff (sometimes they specify a function name that has to be run)
        ds = datasets.load_dataset(
            "parquet",
            data_files = 'https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/test-00000-of-00001.parquet'
        )['train']
        mask_path = Path(__file__).parent / 'taco_mask.txt'
        with open(mask_path) as f:
            mask = f.read()
        mask = ast.literal_eval(mask)
        return [ds[i] for i in mask]
    else:
        raise ValueError(f"Invalid split: {split}")

def format_apps_chat(ds: List[Dict[str, str]], system_prompt: str) -> List[List[Dict[str, str]]]:
    """
    Format the APPS dataset as a list of message lists in chat format.

    Args:
        ds: List of Dictionaries with the following keys: id, question, answer
        system_prompt: System prompt to use for the chat

    Returns:
        List of message lists in chat format
    """
    return [[{"role": "system", "content": system_prompt}, {"role": "user", "content": example['question']}] for example in ds]

async def eval_apps(
    sampling_client: tinker.SamplingClient,
    system_prompt: str,
    apps_prompt: str,
    num_problems: int = 100,
    config: GenerateConfig = None,
    test_timeout: float = 5.0,
    test_max_workers: int = 8,
    split: str = 'apps'
) -> List[Dict]:
    """
    Evaluate a model on the APPS dataset.

    Args:
        sampling_client: Initialized tinker SamplingClient
        system_prompt: System prompt to use for the chat
        apps_prompt: APPS prompt with {problem_statement} placeholder
        num_problems: Number of problems to evaluate on (ignored if mask is provided)
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        test_timeout: Timeout in seconds for each test case
        test_max_workers: Number of parallel workers for testing solutions
        mask: List of problem indices to evaluate on (if None, uses first num_problems)

    Returns:
        List of dicts with problem, expected (test_cases), predicted (code), correct, response
    """
    ds = load_apps_dataset(split = split)[:num_problems]

    # Format messages for generation
    messages_list = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": apps_prompt.format(problem_statement=example['question'])}
        ]
        for example in ds
    ]

    print(f"Evaluating {sampling_client.get_tokenizer().name_or_path} on {len(ds)} APPS problems...")

    # Generate responses
    if config is None:
        config = GenerateConfig()

    outputs = await generate_async(
        sampling_client=sampling_client,
        messages_list=messages_list,
        config=config,
        add_generation_prompt=True,
    )

    # Extract code from responses
    solutions = []
    for output_dict in outputs:
        response = output_dict["output"][0]  # Take first sample
        code = extract_xml_tag(response, 'code')
        if code is None:
            code = ""  # Empty code will fail tests
        solutions.append(code)

    # Extract test cases from dataset
    test_cases_list = []
    for problem in ds:
        test_cases = ast.literal_eval(problem['input_output'])
        test_cases_list.append(test_cases)

    # Test all solutions in parallel
    print(f"Testing {len(solutions)} solutions...")
    test_results = test_solutions_batch(
        solutions=solutions,
        test_cases_list=test_cases_list,
        timeout=test_timeout,
        max_workers=test_max_workers,
    )

    # Build results
    results = []
    num_correct = 0

    for i, (problem, output_dict, code, test_result) in enumerate(zip(ds, outputs, solutions, test_results)):
        response = output_dict["output"][0]
        passed = test_result["passed"]
        reason = test_result["reason"]

        if passed:
            num_correct += 1

        results.append({
            'problem': problem['question'],
            'input': output_dict['input'],
            'correct': passed,
            'reason': reason,
            'response': response,
        })

    accuracy = num_correct / len(ds) if ds else 0.0
    print(f"Accuracy: {num_correct}/{len(ds)} = {accuracy:.2%}")

    return results


APPS_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "apps_prompt.txt"


async def run_apps_evaluation(
    service_client: tinker.ServiceClient,
    paths: List[str],
    system_prompt: str,
    apps_prompt: Optional[str] = None,
    config: GenerateConfig = None,
    num_problems: int = 100,
    test_timeout: float = 5.0,
    test_max_workers: int = 8,
    split: str = 'apps',
    save: bool = True,
    save_dir: str = "logs",
    save_prefix: str = "apps",
) -> tuple:
    """
    Run APPS evaluation on multiple model paths in parallel.

    Returns:
        Tuple of (accuracies, all_results) where:
        - accuracies: List of accuracy floats for each path
        - all_results: List of result lists for each path
    """
    if config is None:
        config = GenerateConfig()

    if apps_prompt is None:
        apps_prompt = APPS_PROMPT_PATH.read_text()

    async def evaluate_path(path: str):
        sampling_client = service_client.create_sampling_client(model_path=path)
        results = await eval_apps(
            sampling_client=sampling_client,
            system_prompt=system_prompt,
            apps_prompt=apps_prompt,
            num_problems=num_problems,
            config=config,
            test_timeout=test_timeout,
            test_max_workers=test_max_workers,
            split=split,
        )
        accuracy = sum(r['correct'] for r in results) / len(results) if results else 0.0

        if save:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{save_prefix}_{path.split('/')[-1]}.json"
            filepath = Path(save_dir) / filename
            with open(filepath, "w") as f:
                import json as _json
                _json.dump(results, f, indent=2)
            print(f'Results stored at {filepath}')

        return accuracy, results

    results = await asyncio.gather(*[evaluate_path(path) for path in paths])
    accuracies = [r[0] for r in results]
    all_results = [r[1] for r in results]
    return accuracies, all_results


async def eval_apps_with_openrouter(
    model: str,
    system_prompt: str,
    apps_prompt: str,
    num_problems: int = 100,
    config: GenerateConfig = None,
    test_timeout: float = 5.0,
    test_max_workers: int = 8,
    split: str = 'apps',
    api_key: str = None,
) -> List[Dict]:
    """
    Evaluate a model on the APPS dataset using OpenRouter.

    Args:
        model: OpenRouter model identifier (e.g. 'qwen/qwen-2.5-7b-instruct')
        system_prompt: System prompt to use for the chat
        apps_prompt: APPS prompt with {problem_statement} placeholder
        num_problems: Number of problems to evaluate on
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples
        test_timeout: Timeout in seconds for each test case
        test_max_workers: Number of parallel workers for testing solutions
        split: Dataset split to use ('apps' or 'taco')
        api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.

    Returns:
        List of dicts with problem, input, correct, reason, response
    """
    if config is None:
        config = GenerateConfig()

    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    ds = load_apps_dataset(split=split)[:num_problems]
    semaphore = asyncio.Semaphore(config.max_concurrent)

    print(f"Evaluating {model} on {len(ds)} APPS problems via OpenRouter...")

    async def generate_single(example, index):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": apps_prompt.format(problem_statement=example['question'])},
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
            "input": messages,
            "response": response.choices[0].message.content,
        }

    tasks = [generate_single(ex, i) for i, ex in enumerate(ds)]
    raw_results = await tqdm_asyncio.gather(*tasks, desc="Generating")

    # Sort by index to maintain order
    raw_results.sort(key=lambda x: x["index"])

    # Extract code from responses
    solutions = []
    for r in raw_results:
        code = extract_xml_tag(r["response"], 'code')
        if code is None:
            code = ""
        solutions.append(code)

    # Extract test cases from dataset
    test_cases_list = []
    for problem in ds:
        test_cases = ast.literal_eval(problem['input_output'])
        test_cases_list.append(test_cases)

    # Test all solutions in parallel
    print(f"Testing {len(solutions)} solutions...")
    test_results = test_solutions_batch(
        solutions=solutions,
        test_cases_list=test_cases_list,
        timeout=test_timeout,
        max_workers=test_max_workers,
    )

    # Build results
    results = []
    num_correct = 0

    for i, (problem, r, code, test_result) in enumerate(zip(ds, raw_results, solutions, test_results)):
        passed = test_result["passed"]
        reason = test_result["reason"]

        if passed:
            num_correct += 1

        results.append({
            'problem': problem['question'],
            'input': r['input'],
            'correct': passed,
            'reason': reason,
            'response': r['response'],
        })

    accuracy = num_correct / len(ds) if ds else 0.0
    print(f"Accuracy: {num_correct}/{len(ds)} = {accuracy:.2%}")

    return results


################
# APPS Scoring #
################

def _run_single_test(args: tuple) -> bool:
    """Helper function to run a single test case (must be at module level for pickling)."""
    import subprocess
    solution_file, test_input, expected_output, timeout = args
    try:
        result = subprocess.run(
            ['python', solution_file],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        actual_output = result.stdout.rstrip('\n')
        expected = expected_output.rstrip('\n')
        return actual_output == expected
    except:
        return False


def test_solution(solution: str, test_cases: Dict[str, List[str]], timeout: float = 5.0) -> bool:
    """
    Test a Python solution against APPS test cases by executing it with stdin/stdout.

    Args:
        solution: Python code as a string that reads from stdin and prints to stdout
        test_cases: Dictionary with 'inputs' and 'outputs' lists of strings
        timeout: Maximum time in seconds to allow for each test case

    Returns:
        True if all test cases pass, False otherwise
    """
    import tempfile
    import os

    inputs = test_cases['inputs']
    outputs = test_cases['outputs']

    # Write solution to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', errors='replace') as f:
        f.write(solution)
        solution_file = f.name

    try:
        # Run all test cases (sequentially - subprocess already spawns processes)
        for test_input, expected_output in zip(inputs, outputs):
            if not _run_single_test((solution_file, test_input, expected_output, timeout)):
                return False
        return True
    finally:
        os.unlink(solution_file)

_BATCH_RUNNER = '''\
import sys, os, io, json, signal, math

def _handler(signum, frame):
    os.write(2, json.dumps({"reason": "timeout"}).encode())
    os._exit(2)

def main():
    solution_path = sys.argv[1]
    test_data_path = sys.argv[2]
    per_case_timeout = max(1, math.ceil(float(sys.argv[3])))

    with open(solution_path) as f:
        code = f.read()
    with open(test_data_path) as f:
        test_data = json.load(f)

    compiled = compile(code, "<solution>", "exec")
    signal.signal(signal.SIGALRM, _handler)

    for inp, exp in zip(test_data["inputs"], test_data["outputs"]):
        try:
            signal.alarm(per_case_timeout)
            sys.stdin = io.StringIO(inp)
            capture = io.StringIO()
            sys.stdout = capture
            exec(compiled, {"__name__": "__main__", "__builtins__": __builtins__})
        except SystemExit:
            pass
        except Exception as e:
            signal.alarm(0)
            sys.stdin = sys.__stdin__
            sys.stdout = sys.__stdout__
            sys.__stderr__.write(json.dumps({
                "reason": "runtime_error",
                "test_case": {"input": inp, "expected": exp},
                "error": str(e)
            }))
            sys.exit(3)
        finally:
            signal.alarm(0)
            sys.stdin = sys.__stdin__
            sys.stdout = sys.__stdout__
        actual = capture.getvalue().rstrip("\\n")
        expected = exp.rstrip("\\n")
        if actual != expected:
            sys.__stderr__.write(json.dumps({
                "reason": "wrong_answer",
                "test_case": {"input": inp, "expected": expected, "actual": actual}
            }))
            sys.exit(1)

main()
'''


def _truncate(s: str, max_len: int = 200) -> str:
    return s if len(s) <= max_len else s[:max_len] + "..."


def _test_solution_all(args: tuple) -> dict:
    """Run all test cases for one solution in a single subprocess.

    Returns dict with 'passed' (bool) and 'reason' (str).
    """
    import subprocess
    import json as _json
    solution_file, test_data_file, per_case_timeout, num_cases, runner_file = args
    try:
        result = subprocess.run(
            [sys.executable, runner_file, solution_file, test_data_file, str(per_case_timeout)],
            capture_output=True,
            text=True,
            timeout=min(per_case_timeout * num_cases + 5, 120),
        )
        if result.returncode == 0:
            return {"passed": True, "reason": "Passed all test cases"}
        # Parse stderr for failure details
        try:
            info = _json.loads(result.stderr)
            reason = info.get("reason", "unknown")
            if reason == "wrong_answer":
                tc = info["test_case"]
                return {
                    "passed": False,
                    "reason": (
                        f"Wrong answer — input: {_truncate(repr(tc['input']))}, "
                        f"expected: {_truncate(repr(tc['expected']))}, "
                        f"got: {_truncate(repr(tc['actual']))}"
                    ),
                }
            elif reason == "timeout":
                return {"passed": False, "reason": "Timeout"}
            elif reason == "runtime_error":
                tc = info["test_case"]
                return {
                    "passed": False,
                    "reason": (
                        f"Runtime error on input: {_truncate(repr(tc['input']))} — "
                        f"{_truncate(info.get('error', 'unknown'))}"
                    ),
                }
            else:
                return {"passed": False, "reason": f"Failed ({reason})"}
        except (_json.JSONDecodeError, KeyError):
            return {"passed": False, "reason": "Failed (unknown reason)"}
    except subprocess.TimeoutExpired:
        return {"passed": False, "reason": "Timeout (process killed)"}
    except Exception:
        return {"passed": False, "reason": "Failed (execution error)"}


def test_solutions_batch(
    solutions: List[str],
    test_cases_list: List[Dict[str, List[str]]],
    timeout: float = 5.0,
    max_workers: Optional[int] = None
) -> List[Dict]:
    """
    Test multiple solutions in parallel.

    Args:
        solutions: List of Python code strings
        test_cases_list: List of test case dictionaries (one per solution)
        timeout: Maximum time in seconds per test case
        max_workers: Number of parallel workers (defaults to CPU count)

    Returns:
        List of dicts with 'passed' (bool) and 'reason' (str) for each solution
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import tempfile
    import os
    import json

    # Write the batch runner script once
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as rf:
        rf.write(_BATCH_RUNNER)
        runner_file = rf.name

    # Write all solutions and test data to temp files
    solution_files = []
    test_data_files = []
    for solution, test_cases in zip(solutions, test_cases_list):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8', errors='replace') as sf:
            sf.write(solution)
            solution_files.append(sf.name)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            json.dump(test_cases, tf)
            test_data_files.append(tf.name)

    try:
        tasks = []
        for sol_file, td_file, test_cases in zip(solution_files, test_data_files, test_cases_list):
            num_cases = len(test_cases.get('inputs', []))
            tasks.append((sol_file, td_file, timeout, num_cases, runner_file))

        results = [{"passed": False, "reason": "Not run"}] * len(solutions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_test_solution_all, task): i
                for i, task in enumerate(tasks)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Testing solutions"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = {"passed": False, "reason": "Failed (executor error)"}
        return results
    finally:
        for f in solution_files + test_data_files + [runner_file]:
            try:
                os.unlink(f)
            except OSError:
                pass
