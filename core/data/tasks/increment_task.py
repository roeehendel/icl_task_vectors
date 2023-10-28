import random
import string
from typing import Iterable, List

from transformers import PreTrainedTokenizer

from core.data.tasks.task import Task


def _unicode_range(start: str, num: int) -> List[str]:
    return [chr(ord(start) + i) for i in range(num)]


def _add_leading_space(strings: List[str]) -> List[str]:
    return [" " + s for s in strings]


_ORDERED_CONCEPTS = {
    # alphabets
    "latin_lower": list(string.ascii_lowercase),
    "latin_upper": list(string.ascii_uppercase),
    # numbers
    "digits": list(string.digits),
    "subscript_digits": _unicode_range("₀", 10),
    "roman_numerals": _add_leading_space(
        [
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
            "VII",
            "VIII",
            "IX",
            "X",
            "XI",
            "XII",
            "XIII",
            "XIV",
            "XV",
            "XVI",
            "XVII",
            "XVIII",
            "XIX",
            "XX",
        ]
    ),
    "months": _add_leading_space(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
    ),
}


class IncrementTask(Task):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        concepts: Iterable[str] = ("latin_lower",),
        increment: int = +1,
    ):
        super().__init__(tokenizer)
        self._increment = increment
        self._concepts = concepts

    def sample_inputs(self, num_inputs: int, exclude: Iterable[str] = ()) -> List[str]:
        input_space = self.input_space
        return random.sample(set(input_space) - set(exclude), num_inputs)

    @property
    def input_space(self) -> List[int]:
        ordered_values = []
        for concept in self._concepts:
            concept_ordered_values = _ORDERED_CONCEPTS[concept]
            if self._increment > 0:
                ordered_values += concept_ordered_values[: -self._increment]
            else:
                ordered_values += concept_ordered_values[-self._increment :]
        return ordered_values

    def calc_output(self, inp) -> int:
        # first identify the concept
        for concept in self._concepts:
            concept_ordered_values = _ORDERED_CONCEPTS[concept]
            if inp in concept_ordered_values:
                break

        # then return the next/prev value in the concept
        return concept_ordered_values[concept_ordered_values.index(inp) + self._increment]

    def num_examples(self) -> int:
        return len(self.input_space)
