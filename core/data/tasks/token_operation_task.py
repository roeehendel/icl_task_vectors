import itertools
import random
import string
from typing import List, Literal, Iterable

from core.data.tasks.task import Task


class TokenOprationTask(Task):
    def __init__(
        self,
        tokenizer,
        operation: Literal["to_upper", "to_lower", "char_to_int", "int_to_char"],
        input_space: List[str] = string.ascii_lowercase,
    ):
        super().__init__(tokenizer)
        self.operation = operation
        self.input_space = input_space

    def sample_inputs(self, num_inputs: int, exclude: List[str] = ()) -> List[str]:
        return random.sample(set(self.input_space) - set(exclude), num_inputs)

    def calc_output(self, inp) -> str:
        if self.operation == "to_upper":
            return inp.upper()
        elif self.operation == "to_lower":
            return inp.lower()
        elif self.operation == "char_to_int":
            # a->1, b->2, c->3, ...
            return str(ord(inp) - ord("a") + 1)
        elif self.operation == "int_to_char":
            # 1->a, 2->b, 3->c, ...
            return chr(ord("a") + int(inp) - 1)

    def num_examples(self) -> int:
        return len(self.input_space)
