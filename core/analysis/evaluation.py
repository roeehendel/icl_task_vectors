from typing import List

import numpy as np

from core.data.datasets.few_shot_dataset import FewShotDataset
from core.data.tasks.task import Task


def calculate_accuracy(task: Task, predictions: List[str], expected_outputs: List[str]) -> List[bool]:
    correct = _evaluate_predictions(task, predictions, expected_outputs)
    accuracy = correct.mean()
    return accuracy


def calculate_accuracy_on_datasets(task: Task, predictions: List[str], datasets: List[FewShotDataset]) -> List[bool]:
    expected_outputs = [dataset.test_output for dataset in datasets]
    return calculate_accuracy(task, predictions, expected_outputs)


def print_evaluation_summary(task: Task, predictions: List[str], datasets: List[FewShotDataset]) -> None:
    expected_outputs = [dataset.test_output for dataset in datasets]
    inputs = [dataset.test_input for dataset in datasets]
    correct = _evaluate_predictions(task, predictions, expected_outputs)
    accuracy = correct.mean()

    print("Out:\t", predictions)
    print("Exp.:\t", expected_outputs)

    print(f"Accuracy: {accuracy:.2f}")
    if accuracy < 1:
        print("Error cases:")

        # print as a table, with header. The column width is the length of the longest string in the column or the header
        headers = ["Input", "Output", "Expected"]

        column_width = (
            max(max(len(str(x)) for x in column) for column in zip(headers, inputs, predictions, expected_outputs)) + 4
        )

        # sort all lists by the input
        inputs, predictions, expected_outputs, correct = zip(
            *sorted(zip(inputs, predictions, expected_outputs, correct), key=lambda x: x[0])
        )

        print(f"{'Input':{column_width}}{'Output':{column_width}}{'Expected':{column_width}}")
        for inp, prediction, expected_output, corr in zip(inputs, predictions, expected_outputs, correct):
            if not corr:
                print(f"{inp:{column_width}}{prediction:{column_width}}{expected_output:{column_width}}")


def _evaluate_predictions(task: Task, predictions: List[str], expected_outputs: List[str]) -> List[bool]:
    predictions = _strip_whitespace(predictions)
    expected_outputs = _strip_whitespace(expected_outputs)

    vectorized_compare = np.vectorize(task.compare_outputs)
    correct = vectorized_compare(predictions, expected_outputs)

    return correct


def _strip_whitespace(lst: List[str]) -> List[str]:
    return [x.strip() for x in lst]


def _compare(prediction: str, expected_output: str) -> bool:
    return prediction == expected_output
