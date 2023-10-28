from dataclasses import asdict
from typing import ContextManager, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from core.data.datasets.few_shot_dataset import FewShotDataset
from core.data.datasets.few_shot_format import FewShotFormat
from core.models.context_managers.tracing.forward_trace import ForwardTrace
from core.models.context_managers.tracing.forward_tracer import ForwardTracer
from core.models.context_managers.utils import CombinedContextManager
from core.models.utils.llm_layers import get_lm_pipeline
from core.utils.misc import get_nested_tensor_size
from core.utils.nested import nested_apply, nested_concat


def traced_forward(
    model: PreTrainedModel,
    inputs: Dict,
    forward_kwargs: Optional[dict] = None,
    batch_size: Optional[int] = None,
    forward_modifiers: Optional[Iterable[ContextManager]] = (),
) -> Tuple[CausalLMOutputWithPast, ForwardTrace]:
    context_manager, forward_trace = traced_forward_context_manager(model)
    with context_manager:
        outputs = modified_forward(
            model,
            inputs=inputs,
            forward_kwargs=forward_kwargs,
            batch_size=batch_size,
            forward_modifiers=forward_modifiers,
        )

    return outputs, forward_trace


def modified_forward(
    model: PreTrainedModel,
    inputs: Dict,
    forward_kwargs: Optional[dict] = None,
    batch_size: Optional[int] = None,
    forward_modifiers: Optional[Iterable[ContextManager]] = (),
) -> CausalLMOutputWithPast:
    context_manager = modified_forward_context_manager(model, forward_modifiers=forward_modifiers)
    with context_manager:
        outputs = batch_forward(
            model,
            inputs=inputs,
            forward_kwargs=forward_kwargs,
            batch_size=batch_size,
        )

    return outputs


def get_input_type(inputs: Dict) -> str:
    if "input_ids" not in inputs and "inputs_embeds" not in inputs:
        raise ValueError("inputs must contain either input_ids or inputs_embeds")
    if "input_ids" in inputs and "inputs_embeds" in inputs:
        raise ValueError("inputs must contain either input_ids or inputs_embeds, not both")

    input_type = "input_ids" if "input_ids" in inputs else "inputs_embeds"

    return input_type


def _get_forward_kwargs(forward_kwargs: Optional[Dict] = None) -> Dict:
    forward_kwargs = forward_kwargs or {}

    # forward_kwargs.setdefault("output_hidden_states", True)
    # forward_kwargs.setdefault("output_attentions", True)

    return forward_kwargs


def _get_batches(inputs: Dict, batch_size: int, show_progress: bool = False) -> Iterable[Dict]:
    input_type = get_input_type(inputs)

    num_inputs = len(inputs[input_type])
    batches_idx = range(0, num_inputs, batch_size)
    batches = (nested_apply(inputs, lambda t: t[i : i + batch_size]) for i in batches_idx)
    if show_progress:
        batches = tqdm(batches)

    return batches


def batch_forward(
    model: PreTrainedModel,
    inputs: Dict,
    forward_kwargs: Optional[Dict] = None,
    batch_size: int = 100,
    show_progress: bool = False,
) -> CausalLMOutputWithPast:
    batch_size = batch_size or _auto_batch_size(model, inputs)
    forward_kwargs = _get_forward_kwargs(forward_kwargs)

    batches = _get_batches(inputs, batch_size, show_progress=show_progress)

    device = model.device

    outputs = []
    for batch_inputs in batches:
        batch_inputs = nested_apply(batch_inputs, lambda t: t.to(device))

        with torch.no_grad():
            out = model(**batch_inputs, **forward_kwargs)
            output_class = out.__class__
            out = nested_apply(out, lambda t: t.cpu())
        outputs.append(out)

    return output_class(**nested_concat(outputs))


def _auto_batch_size(model: PreTrainedModel, inputs: Dict) -> int:
    base_batch_size = 400
    base_model_size_gb = 11.5  # pythia-12b
    base_sequence_length = 50

    model_size_gb = sum(get_nested_tensor_size(t) for t in model.parameters()) / (1024**3)
    sequence_length = inputs[get_input_type(inputs)].shape[1]

    batch_size = int(base_batch_size * (base_model_size_gb / model_size_gb) * (base_sequence_length / sequence_length))

    # print(f"Model size: {model_size_gb:.2f} GB")
    # print(f"Sequence length: {sequence_length}")
    # print(f"Inferred batch size: {batch_size}")

    return batch_size


def batch_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: Dict,
    generate_kwargs: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = False,
) -> List[str]:
    batch_size = batch_size or _auto_batch_size(model, inputs)

    generate_kwargs = _get_forward_kwargs(generate_kwargs)
    batches = _get_batches(inputs, batch_size, show_progress=show_progress)
    input_type = get_input_type(inputs)

    device = model.device

    generate_ids = []
    for batch_inputs in batches:
        batch_inputs = nested_apply(batch_inputs, lambda t: t.to("cpu"))

        batch_ids = model.generate(
            **batch_inputs,
            **generate_kwargs,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        generate_ids.append(batch_ids)

    generate_ids = torch.cat(generate_ids, dim=0)

    new_ids = generate_ids[:, inputs[input_type].shape[1] :]

    # outs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # completions = [out[len(prompt) :] for out, prompt in zip(outs, prompts)]

    return new_ids


def decode_predictions(
    output_ids: torch.Tensor, tokenizer: PreTrainedTokenizer, few_shot_format: FewShotFormat = FewShotFormat()
) -> List[str]:
    new_tokens = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    answers = [tokens.split(few_shot_format.example_separator)[0] for tokens in new_tokens]
    return answers


def tokenize_prompts(tokenizer: PreTrainedTokenizer, prompts: List[str]) -> torch.Tensor:
    return tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False)


def tokenize_datasets(
    tokenizer: PreTrainedTokenizer,
    datasets: List[FewShotDataset],
    few_shot_format: FewShotFormat = FewShotFormat(),
    format_dataset_kwargs: Optional[dict] = {},
) -> torch.Tensor:
    prompts = few_shot_format.format_datasets(datasets, **format_dataset_kwargs)
    return tokenize_prompts(tokenizer, prompts)


def hidden_to_logits(model: PreTrainedModel, hidden: torch.Tensor) -> torch.Tensor:
    device = model.device

    lm_pipeline = get_lm_pipeline(model)

    hidden = hidden.to(device)
    hidden = hidden.type(lm_pipeline.parameters().__next__().dtype)

    with torch.no_grad():
        logits = lm_pipeline(hidden).cpu()

    return logits


def logits_to_tokens(
    logits: torch.Tensor, tokenizer: PreTrainedTokenizer, ignore_ids: Optional[List[int]] = None
) -> List[str]:
    if ignore_ids is not None:
        logits[np.arange(len(logits)), ignore_ids] = -np.inf

    ids = logits.argmax(dim=-1).numpy()
    tokens = np.vectorize(tokenizer.decode)(ids)
    return tokens


def get_logits(
    model: PreTrainedModel,
    forward_trace: ForwardTrace,
    position: int,
    layer: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    layer_indexer = layer if layer is not None else slice(None, None, None)

    logits = {
        name: hidden_to_logits(model, hidden[:, layer_indexer, position])
        for name, hidden in asdict(forward_trace.residual_stream).items()
    }
    return logits


def traced_forward_context_manager(model: PreTrainedModel) -> Tuple[ContextManager, ForwardTrace]:
    forward_trace = ForwardTrace()
    context_manager = ForwardTracer(model, forward_trace)
    return context_manager, forward_trace


def modified_forward_context_manager(
    model: PreTrainedModel, forward_modifiers: Optional[Iterable[ContextManager]] = ()
) -> ContextManager:
    context_manager = CombinedContextManager([*forward_modifiers])
    return context_manager
