# This must be first
from dotenv import load_dotenv

load_dotenv(".env")

import os
import sys
from typing import Any, Optional
import pickle
from transformers import PreTrainedModel, PreTrainedTokenizer

from core.analysis.evaluation import calculate_accuracy
from core.config import RESULTS_DIR
from core.data.task_helpers import get_task_by_name
from core.data.tasks.task import Task
from core.models.llm_loading import load_model_and_tokenizer
from core.task_vectors import run_icl, run_overriding_task_vector
from core.utils.misc import seed_everything


def is_valid_input(task: Task, inp: Any) -> bool:
    try:
        task.calc_output(inp)
        return True
    except:
        return False


OVERRIDING_TASK_PAIRS = [
    ("algorithmic_next_letter", "algorithmic_to_upper"),
    ("algorithmic_list_last", "algorithmic_list_first"),
    ("algorithmic_prev_letter", "algorithmic_next_letter"),
    ("linguistic_present_simple_past_simple", "linguistic_present_simple_gerund"),
    ("translation_en_es", "translation_en_fr"),
]


def run_overriding_experiment_on_task_pair(model, tokenizer, task_name, overriding_task_name):
    seed_everything(41)

    num_examples = 4

    task = get_task_by_name(tokenizer, task_name)
    overriding_task = get_task_by_name(tokenizer, overriding_task_name)

    test_datasets = task.create_datasets(num_datasets=1000, num_examples=num_examples)
    overriding_datasets = overriding_task.create_datasets(num_datasets=100, num_examples=num_examples)

    # filter only test_datasets that are valid inputs for the overriding task
    test_datasets = [dataset for dataset in test_datasets if is_valid_input(overriding_task, dataset.test_input)]
    test_datasets = test_datasets[: len(overriding_datasets)]

    assert len(test_datasets) == len(overriding_datasets)

    icl_predictions = run_icl(model, tokenizer, task, test_datasets)
    tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_overriding_task_vector(
        model,
        tokenizer,
        task,
        test_datasets,
        overriding_datasets,
    )

    expected_outputs_original = [dataset.test_output for dataset in test_datasets]
    expected_outputs_patched = [overriding_task.calc_output(dataset.test_input) for dataset in test_datasets]

    icl_accuracy_original = calculate_accuracy(task, icl_predictions, expected_outputs_original)
    icl_accuracy_patched = calculate_accuracy(task, icl_predictions, expected_outputs_patched)

    tv_accuracy_original = calculate_accuracy(task, tv_predictions, expected_outputs_original)
    tv_accuracy_patched = calculate_accuracy(task, tv_predictions, expected_outputs_patched)

    print(f"ICL accuracy original: {icl_accuracy_original:.2f}")
    print(f"ICL accuracy patched: {icl_accuracy_patched:.2f}")
    print(f"TV accuracy original: {tv_accuracy_original:.2f}")
    print(f"TV accuracy patched: {tv_accuracy_patched:.2f}")

    return {
        "icl_accuracy_original": icl_accuracy_original,
        "icl_accuracy_patched": icl_accuracy_patched,
        "tv_accuracy_original": tv_accuracy_original,
        "tv_accuracy_patched": tv_accuracy_patched,
    }


def get_results_file(model_type: str, model_variant: str) -> str:
    return os.path.join(RESULTS_DIR, "overriding", f"{model_type}_{model_variant}.pkl")


def run_overriding_experiment(
    model_type: str,
    model_variant: str,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
):
    results_file = get_results_file(model_type, model_variant)

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)

    for task_name, overriding_task_name in OVERRIDING_TASK_PAIRS:
        if task_name in results:
            print(f"Skipping {task_name} because it's already in the results")
            continue

        experiment_name = f"{task_name}-{overriding_task_name}"

        print(f"Running experiment on {task_name} and {overriding_task_name}")
        results[experiment_name] = run_overriding_experiment_on_task_pair(
            model, tokenizer, task_name, overriding_task_name
        )

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def main():
    if len(sys.argv) == 3:
        model_type, model_variant = sys.argv[1:]
    else:
        # model_type, model_variant = "pythia", "6.9B"
        model_type, model_variant = "llama", "13B"

    run_overriding_experiment(model_type, model_variant)


if __name__ == "__main__":
    main()
