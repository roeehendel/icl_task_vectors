import os
import random
import json
from typing import Any, List, Dict
from core import config

from core.data.tasks.task import Task
from transformers import PreTrainedTokenizer

MIN_NUM_EXAMPLES = 70


def _is_single_token(tokenizer: PreTrainedTokenizer, token: str) -> bool:
    return len(tokenizer.tokenize(f"!{token}")) == 2  # this is a hack, might not work for all tokenizers


def filter_single_token_outputs(tokenizer: PreTrainedTokenizer, mapping: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in mapping.items() if _is_single_token(tokenizer, v)}


class MappingTask(Task):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mapping_type: str,
        mapping_name: str,
        allow_prefix: bool = False,
    ):
        super().__init__(tokenizer)
        self.mapping_type = mapping_type
        self.mapping_name = mapping_name
        self.allow_prefix = allow_prefix

        mapping_file = os.path.join(config.DATA_DIR, mapping_type, f"{mapping_name}.json")
        with open(mapping_file) as f:
            mapping = json.load(f)

        if allow_prefix:
            self.mapping = mapping
        else:
            num_before_filter = len(mapping)

            mapping_leading_space = {f" {k}": f" {v}" for k, v in mapping.items()}

            filtered_mapping = filter_single_token_outputs(tokenizer, mapping)
            filtered_mapping_leading_space = filter_single_token_outputs(tokenizer, mapping_leading_space)

            if len(filtered_mapping_leading_space) >= 0.7 * len(filtered_mapping):
                self.mapping = filtered_mapping_leading_space
            else:
                self.mapping = filtered_mapping

            if len(self.mapping) < MIN_NUM_EXAMPLES:
                print(
                    f"WARNING: mapping {mapping_name} has only {len(self.mapping)} examples after filtering "
                    f"({num_before_filter} before)"
                )

    def sample_inputs(self, num_inputs: int, exclude: List[str] = ()) -> List[str]:
        input_space = list(self.mapping.keys())
        return random.sample(set(input_space) - set(exclude), num_inputs)

    def calc_output(self, inp) -> str:
        return self.mapping[inp]

    def num_examples(self) -> int:
        return len(self.mapping)

    def compare_outputs(self, output1: Any, output2: Any) -> bool:
        if self.mapping_type == "translation":
            output1, output2 = output1.strip(), output2.strip()
            output_lang = self.mapping_name.split("_")[1]
            synonyms1 = get_synonyms(output1, output_lang)
            synonyms2 = get_synonyms(output2, output_lang)
            return len(set(synonyms1) & set(synonyms2)) > 0

        return super().compare_outputs(output1, output2)
