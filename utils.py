import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

import nest_asyncio
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import datum_from_model_input_weights, compute_mean_nll
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import torch
import random

import matplotlib.pyplot as plt

nest_asyncio.apply()

###################################################
# Everything is Written for Qwen3 Instruct Models #
# It may transfer to other models.                #
###################################################

##############
# Matplotlib #
##############

def set_matplotlib_style():
    """
    Apply ggplot style and force all text/font colors to black.
    """
    plt.style.use("ggplot")

    plt.rcParams.update({
        # Text colors
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.titlecolor": "black",

        # Legend
        "legend.labelcolor": "black",

        # plt.title(...)
        "axes.titlesize": 12,

        # plt.suptitle(...)
        "figure.titlesize": 14,
    })

def compute_bootstrap_diffs(correct_a, correct_b, n_bootstraps = 10000):
    n = len(correct_a)
    bootstrap_indices = np.random.randint(0, n, size = (n_bootstraps, n))
    bootstrap_a = correct_a[bootstrap_indices]
    bootstrap_b = correct_b[bootstrap_indices]
    bootstrap_p_a = bootstrap_a.mean(axis = -1)
    bootstrap_p_b = bootstrap_b.mean(axis = -1)
    return bootstrap_p_b - bootstrap_p_a

###########
# Parsing #
###########

def extract_xml_tag(text: str, tag: str) -> Optional[str]:
    """
    Extract content from an XML tag in the text.

    Args:
        text: The text to search in
        tag: The XML tag name (without angle brackets)

    Returns:
        The content between the opening and closing tags, or None if not found
    """
    opening_tag = f"<{tag}>"
    closing_tag = f"</{tag}>"
    if (opening_tag in text) and (closing_tag in text):
        # get the last opening tag and the first closing tag after it
        return text.split(opening_tag)[-1].split(closing_tag)[0]
    else:
        return None

#############
# Renderers #
#############

# Map from a substring of tokenizer.name_or_path to the tinker-cookbook renderer name.
# Add new model families here.
# in order of specificity
# make sure theyre in lower case
RENDERER_MAP = {
    'qwen/qwen3-30b-a3b': 'qwen3_disable_thinking',
    'qwen/qwen3-8b': 'qwen3_disable_thinking',
    'qwen': 'qwen3_instruct',
    'deepseek': 'deepseekv3_disable_thinking',
    'llama': 'llama3',
}

def get_renderer(tokenizer):
    """Return the tinker-cookbook renderer for a tokenizer's model family."""
    model_name = tokenizer.name_or_path.lower()
    for key, renderer_name in RENDERER_MAP.items():
        if key in model_name:
            return renderers.get_renderer(renderer_name, tokenizer)
    raise ValueError(f"Unknown model family for renderer: {tokenizer.name_or_path}")


# Model families whose generation goes through the tinker renderer instead of
# the tokenizer's chat template (e.g. llama needs knowledge-cutoff handling,
# deepseek needs thinking disabled).
RENDERER_GENERATION_FAMILIES = ('llama', 'deepseek')


def build_generation_input(tokenizer, messages, add_generation_prompt: bool = True) -> str:
    """Build the input text for generation, dispatching by model family."""
    model_name = tokenizer.name_or_path.lower()
    if any(family in model_name for family in RENDERER_GENERATION_FAMILIES):
        if not add_generation_prompt:
            raise ValueError(f"add_generation_prompt must be True for {model_name}")
        prompt = get_renderer(tokenizer).build_generation_prompt(messages)
        return tokenizer.decode(prompt.to_ints())
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


################
# Abstractions #
################

@dataclass
class GenerateConfig:
    """Configuration for generation parameters."""
    temperature: float = 1.0
    max_tokens: int = 10000
    max_concurrent: int = 2000
    num_samples: int = 1
    cache: bool = True


@dataclass
class TrainConfig:
    """Configuration for training parameters for SFT."""
    lr: float = 1e-4
    batch_size: int = 128
    num_epochs: int = 10
    num_examples: int = 100
    save_sampling_step: int = 1
    save_training_step: int = -1
    save_every_n_steps: int = None
    save_training_every_n_steps: int = None

###########
# Caching #
###########

# Cache directory for storing generations
CACHE_DIR = Path(__file__).parent / ".generation_cache"


def _get_cache_key(
    model_id: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    num_samples: int,
) -> str:
    """Generate a unique cache key for a generation request."""
    cache_data = {
        "model_id": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "num_samples": num_samples,
    }
    cache_str = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_str.encode()).hexdigest()


def _load_from_cache(cache_key: str) -> Optional[Dict]:
    """Load cached generation if it exists."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return None
        # Handle old cache format (list of strings) vs new format (dict with input/output)
        if isinstance(data, list):
            return {"input": None, "output": data}
        return data
    return None


def _save_to_cache(cache_key: str, input_text: str, outputs: List[str]) -> None:
    """Save generation to cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w") as f:
        json.dump({"input": input_text, "output": outputs}, f)


########################
# Generate from Tinker #
########################

def get_model_info(sampling_client):
    """Extract model info (base_model, model_path) from a sampling client."""
    from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
    from tinker import types
    
    async def _get_sampler_async():
        async def _send_request():
            with sampling_client.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                return await client.get(
                    f"/api/v1/samplers/{sampling_client._sampling_session_id}",
                    cast_to=types.GetSamplerResponse,
                )
        return await sampling_client.holder.execute_with_retries(_send_request)
    
    return sampling_client.holder.run_coroutine_threadsafe(_get_sampler_async()).result()

async def generate_async(
    sampling_client: tinker.SamplingClient,
    messages_list: List[List[Dict[str, str]]],
    config: GenerateConfig = None,
    add_generation_prompt: bool = True,
    prefill: bool = False,
) -> List[Dict]:
    """
    Generate outputs from a model using asyncio for parallelization.

    Args:
        sampling_client: Initialized tinker SamplingClient
        messages_list: List of message lists in chat format with 'role' and 'content'
        config: GenerateConfig with temperature, max_tokens, max_concurrent, num_samples, cache
        add_generation_prompt: Whether to add generation prompt for chat format
        prefill: Whether assistant has been prefilled

    Returns:
        List of dicts, where each dict contains:
        - "input": the tokenized input text
        - "output": list of generated output strings
    """
    if config is None:
        config = GenerateConfig()

    tokenizer = sampling_client.get_tokenizer()
    info = get_model_info(sampling_client)
    model_id = info.model_path
    if model_id is None:
        model_id = info.base_model

    sampling_params = tinker.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature
    )
    semaphore = asyncio.Semaphore(config.max_concurrent)

    # Pre-compute input_text for ALL items (needed for return value)
    
    all_input_texts = []
    print('Beginning Tokenization...')
    for messages in tqdm(messages_list, desc="Tokenizing"):
        input_text = build_generation_input(tokenizer, messages, add_generation_prompt=add_generation_prompt)
        if prefill:
            prefill_text = messages[-1]['content']
            index = input_text.rfind(prefill_text)
            input_text = input_text[:index + len(prefill_text)]
        all_input_texts.append(input_text)
    # Check cache and prepare inputs
    all_results = [None] * len(messages_list)
    uncached_indices = []
    cache_keys = []

    for i, messages in enumerate(messages_list):
        cache_key = _get_cache_key(model_id, messages, config.max_tokens, config.temperature, config.num_samples)
        cache_keys.append(cache_key)

        if config.cache:
            cached = _load_from_cache(cache_key)
            if cached is not None:
                # Use precomputed input_text (in case cache has None for old format)
                all_results[i] = {"input": all_input_texts[i], "output": cached["output"]}
                continue

        uncached_indices.append(i)

    if uncached_indices:
        print(f"Cache: {len(messages_list) - len(uncached_indices)}/{len(messages_list)} hits, generating {len(uncached_indices)} new ({config.max_concurrent} concurrent requests)")
    else:
        print(f"Cache: {len(messages_list)}/{len(messages_list)} hits, all cached, ({config.max_concurrent} concurrent requests)")
        return all_results

    print('Starting generation...')
    # Pre-tokenize uncached inputs upfront (CPU-bound, do before async)
    model_inputs = []
    uncached_cache_keys = []
    uncached_input_texts = []
    for i in uncached_indices:
        input_text = all_input_texts[i]
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        model_inputs.append(tinker.ModelInput.from_ints(input_ids))
        uncached_cache_keys.append(cache_keys[i])
        uncached_input_texts.append(input_text)

    async def generate_single(model_input: tinker.ModelInput, cache_key: str, input_text: str) -> Dict:
        """Generate outputs for a single pre-tokenized input and cache immediately."""
        async with semaphore:
            result = await sampling_client.sample_async(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=config.num_samples
            )

        # Decode outside semaphore to not block other requests
        outputs = []
        for sequence in result.sequences:
            decoded = tokenizer.decode(sequence.tokens)
            outputs.append(decoded)

        # Cache immediately after generation
        if config.cache:
            _save_to_cache(cache_key, input_text, outputs)

        return {"input": input_text, "output": outputs}

    print('Finished tokenization, starting generation...')
    # Create all tasks and run with semaphore-limited concurrency
    tasks = [generate_single(model_input, ck, inp) for model_input, ck, inp in zip(model_inputs, uncached_cache_keys, uncached_input_texts)]
    generated_results = await tqdm_asyncio.gather(*tasks, desc="Generating")

    # Merge results (caching already done in generate_single)
    for i, idx in enumerate(uncached_indices):
        all_results[idx] = generated_results[i]

    return all_results


################
# SFT Training #
################

@dataclass
class SFTExample:
    """A single SFT training example with chat message input and output."""
    input: List[Dict[str, str]]  # Chat messages for the prompt
    output: List[Dict[str, str]]  # Chat messages for the completion

def _compute_mean_nll(fwdbwd_result, batch_datums) -> float:
    """Compute mean NLL from forward_backward result and batch data."""
    logprobs = [out['logprobs'] for out in fwdbwd_result.loss_fn_outputs]
    weights = [d.loss_fn_inputs['weights'] for d in batch_datums]
    return compute_mean_nll(logprobs, weights)


def sft_train(
    training_client: tinker.TrainingClient,
    data: List[SFTExample],
    config: TrainConfig = None,
    run_name: str = '',
    shuffle: bool = True,
) -> Dict:
    """
    Vanilla supervised fine-tuning on a list of SFTExamples.

    Args:
        training_client: Tinker TrainingClient for training
        data: List of SFTExample with input and output messages
        config: TrainConfig with lr, batch_size, num_epochs, save_sampling_step, save_training_step
        run_name: Name prefix for saved checkpoints

    Returns:
        Dictionary with:
        - losses: List of all batch losses
        - num_steps: Total number of training steps
        - avg_loss: Average loss across all steps
        - sampling_paths: List of saved checkpoint paths
    """
    if config is None:
        config = TrainConfig()

    print(f"SFT Training: Learning rate: {config.lr}, Batch size: {config.batch_size}, Epochs: {config.num_epochs}")

    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(tokenizer)
    all_losses = []
    sampling_paths = []
    training_paths = []

    sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_epoch_{0}").result().path
    sampling_paths.append(sampling_path)
    print(f"Saved sampling checkpoint: {sampling_path}")

    save_by_steps = config.save_every_n_steps is not None
    save_training_by_steps = config.save_training_every_n_steps is not None
    if save_by_steps:
        print(f'Beginning SFT training on {len(data)} examples for {config.num_epochs} epoch(s) (saving every {config.save_every_n_steps} steps)...')
    else:
        print(f'Beginning SFT training on {len(data)} examples for {config.num_epochs} epochs...')

    data_to_write = []
    global_step = 0

    random.seed(42)

    for epoch in range(1, config.num_epochs + 1):
        # shuffle data every epoch
        if shuffle:
            random.shuffle(data)
        print(f"\n=== Epoch {epoch}/{config.num_epochs} ===")

        epoch_losses = []
        batch_datums = []

        pbar = tqdm(data, desc=f"Training epoch {epoch}/{config.num_epochs}")

        for sft_example in pbar:
            messages = sft_example.input + sft_example.output
            model_input, weights = renderer.build_supervised_example(messages)

            if epoch == 1:
                # only write once — extract tokens from ModelInput chunks for logging
                all_tokens = []
                for chunk in model_input.chunks:
                    if isinstance(chunk, tinker.types.EncodedTextChunk):
                        all_tokens.extend(chunk.tokens)
                    else:
                        all_tokens.extend([0] * chunk.length)
                tokens_tensor = torch.tensor(all_tokens)
                example_to_write = {
                    'gradients': tokenizer.decode(tokens_tensor[weights.bool()]),
                    'no_gradients': tokenizer.decode(tokens_tensor[~weights.bool()])
                }
                data_to_write.append(example_to_write)

            datum = datum_from_model_input_weights(model_input, weights)

            batch_datums.append(datum)

            if len(batch_datums) >= config.batch_size:
                fwdbwd_future = training_client.forward_backward(batch_datums, "cross_entropy")
                optim_future = training_client.optim_step(tinker.AdamParams(learning_rate=config.lr))
                loss = _compute_mean_nll(fwdbwd_future.result(), batch_datums)
                optim_future.result()

                all_losses.append(loss)
                epoch_losses.append(loss)
                batch_datums = []
                global_step += 1
                pbar.set_postfix({"loss": f"{loss:.4f}", "step": global_step})

                # Save by gradient steps
                if save_by_steps and global_step % config.save_every_n_steps == 0:
                    sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_step_{global_step}").result().path
                    sampling_paths.append(sampling_path)
                    print(f"Saved sampling checkpoint (step {global_step}): {sampling_path}")

                if save_training_by_steps and global_step % config.save_training_every_n_steps == 0:
                    training_path = training_client.save_state(name=f"{run_name}_step_{global_step}").result().path
                    training_paths.append(training_path)
                    print(f"Saved training checkpoint (step {global_step}): {training_path}")

        # Flush last partial batch
        if batch_datums:
            fwdbwd_future = training_client.forward_backward(batch_datums, "cross_entropy")
            optim_future = training_client.optim_step(tinker.AdamParams(learning_rate=config.lr))
            loss = _compute_mean_nll(fwdbwd_future.result(), batch_datums)
            optim_future.result()
            all_losses.append(loss)
            epoch_losses.append(loss)
            global_step += 1

            # Save by gradient steps when num_epochs == 1
            if save_by_steps and global_step % config.save_every_n_steps == 0:
                sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_step_{global_step}").result().path
                sampling_paths.append(sampling_path)
                print(f"Saved sampling checkpoint (step {global_step}): {sampling_path}")

            if save_training_by_steps and global_step % config.save_training_every_n_steps == 0:
                training_path = training_client.save_state(name=f"{run_name}_step_{global_step}").result().path
                training_paths.append(training_path)
                print(f"Saved training checkpoint (step {global_step}): {training_path}")

        epoch_avg = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch {epoch} average loss: {epoch_avg:.4f}")

        # Save checkpoint every save_step epochs (or on final epoch), unless saving by steps
        if not save_by_steps and (epoch % config.save_sampling_step == 0 or epoch == config.num_epochs):
            sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_epoch_{epoch}").result().path
            sampling_paths.append(sampling_path)
            print(f"Saved sampling checkpoint: {sampling_path}")

        # Save training checkpoint if save_training_step is positive (epoch-based)
        if not save_training_by_steps and config.save_training_step > 0:
            if epoch % config.save_training_step == 0 or epoch == config.num_epochs:
                training_path = training_client.save_state(name=f"{run_name}_epoch_{epoch}").result().path
                training_paths.append(training_path)
                print(f"Saved training checkpoint: {training_path}")

    # Always save final checkpoint if it wasn't already saved
    if save_by_steps and (global_step % config.save_every_n_steps != 0):
        sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_step_{global_step}").result().path
        sampling_paths.append(sampling_path)
        print(f"Saved final sampling checkpoint (step {global_step}): {sampling_path}")

    if save_training_by_steps and (global_step % config.save_training_every_n_steps != 0):
        training_path = training_client.save_state(name=f"{run_name}_step_{global_step}").result().path
        training_paths.append(training_path)
        print(f"Saved final training checkpoint (step {global_step}): {training_path}")

    return {
        "losses": all_losses,
        "num_steps": len(all_losses),
        "avg_loss": sum(all_losses) / len(all_losses) if all_losses else 0.0,
        "sampling_paths": sampling_paths,
        "training_paths": training_paths,
        "training_data": data_to_write,
    }

def weird_sft_train(
    training_client: tinker.TrainingClient,
    data: List[SFTExample],
    config: TrainConfig = None,
    run_name: str = '',
) -> Dict:
    """
    SFT variant with loss = (initial_loss - ce_loss)^2 and true gradients via forward_backward_custom.

    The first microbatch computes CE loss and stores it as initial_loss (constant, no weight update).
    All subsequent microbatches use forward_backward_custom to compute and backprop through
    (initial_loss - mean_nll)^2, giving the correct gradient: -2(L0 - L) * dL/dθ.
    """
    if config is None:
        config = TrainConfig()

    print(f"SFT Training: Learning rate: {config.lr}, Batch size: {config.batch_size}, Epochs: {config.num_epochs}")

    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(tokenizer)
    all_losses = []
    sampling_paths = []
    training_paths = []

    sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_epoch_{0}").result().path
    sampling_paths.append(sampling_path)
    print(f"Saved sampling checkpoint: {sampling_path}")

    save_by_steps = config.save_every_n_steps is not None
    save_training_by_steps = config.save_training_every_n_steps is not None
    if save_by_steps:
        print(f'Beginning SFT training on {len(data)} examples for {config.num_epochs} epoch(s) (saving every {config.save_every_n_steps} steps)...')
    else:
        print(f'Beginning SFT training on {len(data)} examples for {config.num_epochs} epochs...')

    data_to_write = []
    global_step = 0
    initial_loss = None

    def _weird_loss_fn(data: list, logprobs: list) -> tuple:
        """Custom loss: (initial_loss - mean_nll)^2 with proper gradients."""
        # Compute mean NLL (same as cross_entropy but in torch)
        total_nll = torch.tensor(0.0)
        total_tokens = 0
        for datum, lp in zip(data, logprobs):
            weights = datum.loss_fn_inputs['weights'].to_torch()
            masked_lp = lp * weights
            total_nll = total_nll - masked_lp.sum()
            total_tokens += weights.sum()
        mean_nll = total_nll / total_tokens
        loss = (initial_loss - mean_nll) ** 2
        return loss, {"weird_loss": loss.item(), "ce_loss": mean_nll.item()}

    def _process_batch(batch_datums):
        nonlocal initial_loss, global_step
        if initial_loss is None:
            # First microbatch: compute CE loss to get initial_loss, no weight update
            fwdbwd_result = training_client.forward_backward(batch_datums, "cross_entropy").result()
            initial_loss = _compute_mean_nll(fwdbwd_result, batch_datums)
            # Discard gradients — initial_loss is a constant
            training_client.optim_step(tinker.AdamParams(learning_rate=0.0)).result()
            weird_loss = 0.0
            ce_loss = initial_loss
            print(f"Initial loss (constant): {initial_loss:.4f}")
        else:
            # Use forward_backward_custom for true (L0 - L)^2 gradients
            fwdbwd_output = training_client.forward_backward_custom(batch_datums, _weird_loss_fn).result()
            training_client.optim_step(tinker.AdamParams(learning_rate=config.lr)).result()
            metrics = fwdbwd_output.metrics
            weird_loss = metrics["weird_loss"]
            ce_loss = metrics["ce_loss"]
        global_step += 1
        return weird_loss, ce_loss

    for epoch in range(1, config.num_epochs + 1):
        # shuffle data every epoch
        if shuffle:
            random.shuffle(data)
        print(f"\n=== Epoch {epoch}/{config.num_epochs} ===")

        epoch_losses = []
        batch_datums = []

        pbar = tqdm(data, desc=f"Training epoch {epoch}/{config.num_epochs}")

        for sft_example in pbar:
            messages = sft_example.input + sft_example.output
            model_input, weights = renderer.build_supervised_example(messages)

            if epoch == 1:
                # only write once — extract tokens from ModelInput chunks for logging
                all_tokens = []
                for chunk in model_input.chunks:
                    if isinstance(chunk, tinker.types.EncodedTextChunk):
                        all_tokens.extend(chunk.tokens)
                    else:
                        all_tokens.extend([0] * chunk.length)
                tokens_tensor = torch.tensor(all_tokens)
                example_to_write = {
                    'gradients': tokenizer.decode(tokens_tensor[weights.bool()]),
                    'no_gradients': tokenizer.decode(tokens_tensor[~weights.bool()])
                }
                data_to_write.append(example_to_write)

            datum = datum_from_model_input_weights(model_input, weights)

            batch_datums.append(datum)

            if len(batch_datums) >= config.batch_size:
                weird_loss, ce_loss = _process_batch(batch_datums)

                all_losses.append(weird_loss)
                epoch_losses.append(weird_loss)
                batch_datums = []
                pbar.set_postfix({"weird_loss": f"{weird_loss:.4f}", "ce_loss": f"{ce_loss:.4f}", "step": global_step})

                # Save by gradient steps
                if save_by_steps and global_step % config.save_every_n_steps == 0:
                    sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_step_{global_step}").result().path
                    sampling_paths.append(sampling_path)
                    print(f"Saved sampling checkpoint (step {global_step}): {sampling_path}")

                if save_training_by_steps and global_step % config.save_training_every_n_steps == 0:
                    training_path = training_client.save_state(name=f"{run_name}_step_{global_step}").result().path
                    training_paths.append(training_path)
                    print(f"Saved training checkpoint (step {global_step}): {training_path}")

        # Flush last partial batch
        if batch_datums:
            weird_loss, ce_loss = _process_batch(batch_datums)

            all_losses.append(weird_loss)
            epoch_losses.append(weird_loss)

            # Save by gradient steps when num_epochs == 1
            if save_by_steps and global_step % config.save_every_n_steps == 0:
                sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_step_{global_step}").result().path
                sampling_paths.append(sampling_path)
                print(f"Saved sampling checkpoint (step {global_step}): {sampling_path}")

            if save_training_by_steps and global_step % config.save_training_every_n_steps == 0:
                training_path = training_client.save_state(name=f"{run_name}_step_{global_step}").result().path
                training_paths.append(training_path)
                print(f"Saved training checkpoint (step {global_step}): {training_path}")

        epoch_avg = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch {epoch} average weird loss: {epoch_avg:.4f}")

        # Save checkpoint every save_step epochs (or on final epoch), unless saving by steps
        if not save_by_steps and (epoch % config.save_sampling_step == 0 or epoch == config.num_epochs):
            sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_epoch_{epoch}").result().path
            sampling_paths.append(sampling_path)
            print(f"Saved sampling checkpoint: {sampling_path}")

        # Save training checkpoint if save_training_step is positive (epoch-based)
        if not save_training_by_steps and config.save_training_step > 0:
            if epoch % config.save_training_step == 0 or epoch == config.num_epochs:
                training_path = training_client.save_state(name=f"{run_name}_epoch_{epoch}").result().path
                training_paths.append(training_path)
                print(f"Saved training checkpoint: {training_path}")

    # Always save final checkpoint if it wasn't already saved
    if save_by_steps and (global_step % config.save_every_n_steps != 0):
        sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_step_{global_step}").result().path
        sampling_paths.append(sampling_path)
        print(f"Saved final sampling checkpoint (step {global_step}): {sampling_path}")

    if save_training_by_steps and (global_step % config.save_training_every_n_steps != 0):
        training_path = training_client.save_state(name=f"{run_name}_step_{global_step}").result().path
        training_paths.append(training_path)
        print(f"Saved final training checkpoint (step {global_step}): {training_path}")

    return {
        "losses": all_losses,
        "num_steps": len(all_losses),
        "avg_loss": sum(all_losses) / len(all_losses) if all_losses else 0.0,
        "initial_loss": initial_loss,
        "sampling_paths": sampling_paths,
        "training_paths": training_paths,
        "training_data": data_to_write,
    }

async def generate_logprobs_async(
    sampling_client: tinker.SamplingClient,
    data: List[Dict],
    max_concurrent: int = 2000,
) -> List[Dict]:
    """
    Compute logprobs for output tokens given input-output message pairs.

    Args:
        sampling_client: Initialized tinker SamplingClient
        data: List of dicts, each with:
              - "input": List of chat messages (the prompt)
              - "output": List of chat messages (must be exactly length 1)
        max_concurrent: Max concurrent requests

    Returns:
        List of dicts, each with:
        - "sum_logprob": float, sum of logprobs for the output tokens
        - "logprobs": List[float], individual per-token logprobs for the output
        - "input": the original input messages
        - "output": the original output messages
    """
    tokenizer = sampling_client.get_tokenizer()
    semaphore = asyncio.Semaphore(max_concurrent)

    # Validate and pre-tokenize
    tokenized = []
    print("Tokenizing for logprobs...")
    for i, entry in enumerate(tqdm(data, desc="Tokenizing")):
        if len(entry['output']) != 1:
            raise ValueError(
                f"Entry {i}: output must have exactly 1 message, got {len(entry['output'])}"
            )

        full_messages = entry['input'] + entry['output']

        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        # Strip trailing end-of-turn token (treat output as prefill)
        idx = full_text.rfind('<|im_end|>')
        if idx != -1:
            full_text = full_text[:idx]

        prompt_text = tokenizer.apply_chat_template(
            entry['input'], tokenize=False, add_generation_prompt=True
        )

        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        n_prompt = len(prompt_ids)

        tokenized.append((full_ids, n_prompt))

    # Compute logprobs in parallel
    async def compute_single(full_ids, n_prompt):
        model_input = tinker.ModelInput.from_ints(full_ids)
        async with semaphore:
            all_logprobs = await sampling_client.compute_logprobs_async(model_input)

        output_logprobs = [lp for lp in all_logprobs[n_prompt:] if lp is not None]
        return {
            'sum_logprob': sum(output_logprobs),
            'logprobs': output_logprobs,
        }

    tasks = [compute_single(*t) for t in tokenized]
    logprob_results = await tqdm_asyncio.gather(*tasks, desc="Computing logprobs")

    # Attach original input/output to each result
    results = []
    for entry, lr in zip(data, logprob_results):
        lr['input'] = entry['input']
        lr['output'] = entry['output']
        results.append(lr)

    avg_sum = sum(r['sum_logprob'] for r in results) / len(results)
    print(f"Computed logprobs for {len(results)} examples (avg sum logprob: {avg_sum:.4f})")

    return results

################
# RL Training  #
################

def rl_train(
    training_client: tinker.TrainingClient,
    sampling_client: tinker.SamplingClient,
    dataset: List,
    format_fn: Callable[[any], List[Dict[str, str]]],
    value_fn: Callable[[str, any], float],
    config: GenerateConfig = None,
    learning_rate: float = 1e-5,
    batch_size: Optional[int] = None,
    save_step: int = 1,
    run_name: str = "rl_run",
    service_client: Optional[tinker.ServiceClient] = None,
    add_generation_prompt: bool = True,
    normalize_advantages_by_length: bool = False,
    center_rewards: bool = True,
) -> Dict:
    """
    RL training using GRPO-style importance sampling with reward centering.

    For each input, samples multiple completions, computes rewards via value_fn,
    centers rewards to get advantages, and trains using importance sampling.

    Set center_rewards=False to skip the group-mean subtraction so groups with
    uniformly correct/incorrect answers still drive a gradient (raw REINFORCE).

    Args:
        training_client: Tinker TrainingClient for training
        sampling_client: Tinker SamplingClient for generating completions
        dataset: List of data items (e.g., olympiad problems with ground truth answers)
        format_fn: Function that converts a data item into a list of messages in chat format
        value_fn: Function that takes (sampling_client, completion_text, data_item) and returns a reward
        config: GenerateConfig with temperature, max_tokens, num_samples, etc.
        learning_rate: Learning rate for Adam optimizer
        batch_size: Number of datums per gradient update. If None, updates on all datums at once.
        save_step: Save checkpoint every N batches
        run_name: Name prefix for saved checkpoints
        service_client: Optional ServiceClient for reloading sampling client with updated weights
        add_generation_prompt: Whether to add generation prompt for chat format
        normalize_advantages_by_length: If True, divide advantages by the number of completion tokens
            so that each datum contributes equally to the gradient regardless of sequence length.

    Returns:
        Dictionary with rewards, avg_reward, num_datums, sampling_paths, and optim_metrics
    """
    if config is None:
        config = GenerateConfig()

    print(f"RL Training (GRPO): lr={learning_rate}, group_size={config.num_samples}, dataset_size={len(dataset)}")

    tokenizer = training_client.get_tokenizer()
    sampling_params = tinker.SamplingParams(max_tokens=config.max_tokens, temperature=config.temperature)
    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    all_rewards = []
    sampling_paths = []

    # Tokenize all prompts and kick off sampling futures
    futures = []
    prompts = []
    data_items = []

    for data_item in dataset:
        messages = format_fn(data_item)

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        data_item['input_text'] = input_text

        prompt_ids = tokenizer.encode(input_text, add_special_tokens=False)
        model_input = tinker.ModelInput.from_ints(prompt_ids)
        prompts.append(model_input)
        data_items.append(data_item)
        futures.append(sampling_client.sample(prompt=model_input, num_samples=config.num_samples, sampling_params=sampling_params))

    # Process results and build datums
    datums = []

    # ground truth rewards for tracking progress
    gt_rewards = []
    def gt_reward_fn(sampling_client, completion: str, data_item) -> float:
        if 'input_output' in data_item:
            return None
        predicted = extract_xml_tag(completion, 'answer')
        if predicted is not None:
            predicted = predicted.strip()
        expected = data_item['answer'].strip()
        return 1.0 if predicted == expected else 0.0

    for prompt, data_item, future in tqdm(zip(prompts, data_items, futures), total=len(futures), desc="Sampling & scoring"):
        sample_result = future.result()

        rewards_G = []
        sampled_tokens_G = []
        logprobs_G = []
        for seq in sample_result.sequences:
            completion_text = tokenizer.decode(seq.tokens)
            reward = value_fn(sampling_client, completion_text, data_item)
            rewards_G.append(reward)
            sampled_tokens_G.append(seq.tokens)
            logprobs_G.append(seq.logprobs)

            # ground truth reward for tracking progress
            try:
                gt_reward = gt_reward_fn(sampling_client, completion_text, data_item)
                if gt_reward is not None:
                    gt_rewards.append(gt_reward)
            except Exception as e:
                pass

        # Center rewards to get advantages (GRPO-style)
        mean_reward = sum(rewards_G) / len(rewards_G)
        if center_rewards:
            advantages_G = [r - mean_reward for r in rewards_G]
        else:
            advantages_G = list(rewards_G)
        all_rewards.append(mean_reward)

        # Skip if all advantages are zero (no learning signal)
        # if all(adv == 0.0 for adv in advantages_G):
        #     continue

        # Build datums for each sample
        ob_len = prompt.length - 1
        for tokens, logprobs, advantage in zip(sampled_tokens_G, logprobs_G, advantages_G):
            model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
            target_tokens = [0] * ob_len + list(tokens)
            padded_logprobs = [0.0] * ob_len + list(logprobs)
            num_completion_tokens = model_input.length - ob_len
            if normalize_advantages_by_length and num_completion_tokens > 0:
                token_advantage = advantage / num_completion_tokens
            else:
                token_advantage = advantage
            padded_advantages = [0.0] * ob_len + [token_advantage] * num_completion_tokens

            datum = tinker.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": tinker.TensorData.from_torch(torch.tensor(padded_logprobs)),
                    "advantages": tinker.TensorData.from_torch(torch.tensor(padded_advantages)),
                },
            )
            datums.append(datum)
    if len(gt_rewards) > 0:
        print(f"Ground truth rewards (only olympiads): {sum(gt_rewards) / len(gt_rewards)}, n = {len(gt_rewards)}")
    else:
        print("No ground truth rewards")

    # Training step(s)
    print(f"Training on {len(datums)} datums...")

    if batch_size is None or batch_size >= len(datums):
        fwdbwd_future = training_client.forward_backward(datums, loss_fn="importance_sampling")
        optim_future = training_client.optim_step(adam_params)
        fwdbwd_future.result()
        optim_result = optim_future.result()
        num_batches = 1
    else:
        num_batches = (len(datums) + batch_size - 1) // batch_size
        optim_result = None

        for batch_idx in tqdm(range(0, len(datums), batch_size), desc="Training batches"):
            batch_datums = datums[batch_idx:batch_idx + batch_size]
            fwdbwd_future = training_client.forward_backward(batch_datums, loss_fn="importance_sampling")
            optim_future = training_client.optim_step(adam_params)
            fwdbwd_future.result()
            optim_result = optim_future.result()

    # Save checkpoint
    sampling_path = training_client.save_weights_for_sampler(name=f"{run_name}_final").result().path
    sampling_paths.append(sampling_path)
    print(f"Saved checkpoint: {sampling_path}, trained in {num_batches} batch(es)")

    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    print(f"Average reward: {avg_reward:.4f}")

    return {
        "rewards": all_rewards,
        "avg_reward": avg_reward,
        "num_datums": len(datums),
        "sampling_paths": sampling_paths,
        "optim_metrics": optim_result.metrics if optim_result and optim_result.metrics else {},
    }