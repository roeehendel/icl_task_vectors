from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE
from core.models.llm_loading import load_tokenizer
from core.data.task_helpers import get_task_by_name

NUM_EXAMPLES_WARNING = 90

tokenizers = [load_tokenizer(*model_args) for model_args in (MODELS_TO_EVALUATE)]
tokenizer_hashes = [hash("".join(str(x) for x in sorted(tokenizer.get_vocab().keys()))) for tokenizer in tokenizers]
print("Total number of tokenizers:", len(tokenizers))
tokenizers = list({hash: tokenizer for hash, tokenizer in zip(tokenizer_hashes, tokenizers)}.values())
print("Unique tokenizers:", len(tokenizers))

bad_tasks = []
for tokenizer in tokenizers:
    print(tokenizer.name_or_path)
    for task_name in TASKS_TO_EVALUATE:
        task = get_task_by_name(tokenizer, task_name)
        print(f"\tTask: {task_name}, num examples: {task.num_examples()}")
        if task.num_examples() < NUM_EXAMPLES_WARNING and "algorithmic" not in task_name:
            bad_tasks.append(tokenizer.name_or_path + "_" + task_name + ": " + str(task.num_examples()))

print("Bad tasks:")
print(bad_tasks)
