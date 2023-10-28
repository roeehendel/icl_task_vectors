import itertools
import random
import string
from typing import List, Literal, Iterable

from core.data.tasks.task import Task


class ListOperationTask(Task):
    def __init__(
        self,
        tokenizer,
        operation: Literal["min", "max", "first", "last", "length"],
        list_lenghts: Iterable[int],
        elements_space: List[str] = string.ascii_lowercase,
        separator: str = ",",
    ):
        super().__init__(tokenizer)
        self.operation = operation
        self.list_lenghts = list_lenghts
        self.separator = separator
        self.elements_space = elements_space

    def _encode_list(self, inp: str) -> str:
        return self.separator.join(inp)

    def _decode_list(self, inp: str) -> str:
        return inp.split(self.separator)

    def _random_input(self) -> str:
        length = random.choice(self.list_lenghts)
        return self._encode_list(random.choices(self.elements_space, k=length))

    def sample_inputs(self, num_inputs: int, exclude: List[str] = ()) -> List[str]:
        # Don't use this code in production ;)
        input_space_size = sum(len(self.elements_space) ** length for length in self.list_lenghts)
        assert (input_space_size - len(exclude)) >= num_inputs, "Not enough inputs to sample from"

        inputs = []
        while len(inputs) < num_inputs:
            inp = self._random_input()
            if inp not in exclude and inp not in inputs:
                inputs.append(inp)
        return inputs

    def calc_output(self, inp) -> str:
        input_list = self._decode_list(inp)

        if self.operation == "min":
            return min(input_list)
        elif self.operation == "max":
            return max(input_list)
        elif self.operation == "first":
            return input_list[0]
        elif self.operation == "last":
            return input_list[-1]
        elif self.operation == "length":
            return len(input_list)

    def num_examples(self) -> int:
        num_examples = 0
        for length in self.list_lenghts:
            num_examples += len(self.elements_space) ** length
        return num_examples
