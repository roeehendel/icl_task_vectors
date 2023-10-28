from typing import List, Optional, Dict

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from core.models.utils.llm_layers import get_attention_layers
from core.data.datasets.few_shot_dataset import FewShotDataset


class AttentionMasker:
    def __init__(self, model: PreTrainedModel, attention_masks: Optional[torch.Tensor] = None):
        self._model = model
        self._attention_masks = attention_masks

        self._attn_layers = get_attention_layers(model)

        self._hooks = []

    def __enter__(self):
        self._register_forward_hooks()

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()

    def _register_forward_hooks(self):
        if self._attention_masks is None:
            return

        def attention_masking_hook(layer_num: int):
            def hook(module, args, kwargs):
                attention_mask = self._attention_masks[layer_num].to(kwargs["attention_mask"].device)

                current_attention_mask = kwargs["attention_mask"].clone()

                not_masked = current_attention_mask[0, 0, 0, 0]
                masked = current_attention_mask[current_attention_mask != not_masked][0]

                current_attention_mask[:, :, attention_mask] = masked

                kwargs["attention_mask"] = current_attention_mask

                return args, kwargs

            return hook

        for i, attn in enumerate(self._attn_layers):
            hook = attn.register_forward_pre_hook(attention_masking_hook(i), with_kwargs=True)
            self._hooks.append(hook)


def few_shot_attention_masks(
    datasets: List[FewShotDataset], tokenizer: PreTrainedTokenizer, verbose: bool = False
) -> Dict[str, torch.Tensor]:
    prompt = datasets[0].prompt
    few_shot_format = datasets[0].few_shot_format

    sequence_length = len(tokenizer(prompt).input_ids)
    num_tokens = len(tokenizer.tokenize(prompt))

    start_idx = sequence_length - num_tokens

    example_length = len(tokenizer.tokenize(prompt.split(few_shot_format.example_separator)[0])) + 1
    num_examples = len(prompt.split(few_shot_format.example_separator)) - 1

    context_length = example_length * num_examples

    context_end_idx = start_idx + context_length
    last_idx = start_idx + num_tokens - 1
    input_idx = last_idx - 1

    context_idx = slice(start_idx, context_end_idx)
    last_example_idx = slice(context_end_idx, last_idx)

    if verbose:
        # sanity check
        tokenized_prompt = [None] * start_idx + tokenizer.tokenize(prompt)
        print("context:", tokenized_prompt[context_idx])
        print("last example:", tokenized_prompt[last_example_idx])
        print("last token:", tokenized_prompt[last_idx])
        print("input token:", tokenized_prompt[input_idx])

    mask_nothing = torch.zeros(sequence_length, sequence_length, dtype=torch.bool)

    mask_context = torch.zeros(sequence_length, sequence_length, dtype=torch.bool)
    mask_context[context_end_idx:, context_idx] = True

    mask_last_example = torch.zeros(sequence_length, sequence_length, dtype=torch.bool)
    mask_last_example[last_idx, last_example_idx] = True

    mask_input = torch.zeros(sequence_length, sequence_length, dtype=torch.bool)
    mask_input[last_idx, input_idx] = True

    mask_all = torch.zeros(sequence_length, sequence_length, dtype=torch.bool)
    mask_all[last_idx, start_idx:last_idx] = True

    return {
        "nothing": mask_nothing,
        "context": mask_context,
        "last_example": mask_last_example,
        "input": mask_input,
        "all": mask_all,
    }
