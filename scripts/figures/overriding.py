import os
import pandas as pd
from scripts.figures.helpers import load_overriding_results
from core.config import FIGURES_DIR


def create_overriding_results_table(overriding_results: dict, model_name: str):
    model_overriding_results = overriding_results[model_name]

    # create a dataframe with the results
    df = pd.DataFrame.from_dict(model_overriding_results, orient="index")

    print(df)

    # remove first part of task name
    df.index = df.index.to_series().apply(lambda x: x.split("_", 1)[1])

    # Round the values to 2 decimal places
    df = df.applymap(lambda x: "{0:.2f}".format(x) if isinstance(x, (int, float)) else x)

    # Change column and index names to be more readable
    df.rename(
        columns={
            "icl_accuracy_original": "ICL - Demonstrations",
            "icl_accuracy_patched": "ICL - Overriding",
            "tv_accuracy_original": "TV - Demonstrations",
            "tv_accuracy_patched": "TV - Overriding",
        },
        index={
            # Replace "_" with " " and capitalize the first letter of each word
            key: key.replace("_", " ").title()
            for key in df.index
        },
        inplace=True,
    )

    latex_table = df.to_latex()

    # save the table to a file
    save_path = os.path.join(FIGURES_DIR, f"overriding_results_{model_name}.tex")
    with open(save_path, "w") as f:
        f.write(latex_table)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    model_name = "llama_13B"

    overriding_results = load_overriding_results()

    create_overriding_results_table(overriding_results, model_name)


if __name__ == "__main__":
    main()
