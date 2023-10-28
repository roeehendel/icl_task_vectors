import requests
import json
import os
from core import config


RELATIONS_TO_INCLUDE = {
    "P36": "country_capital",
    "P30": "location_continent",
    "P131": "location_country",
    "P37": "location_language",
    "P140": "location_religion",
    "P1412": "person_language",
    "P106": "person_profession",
    "P413": "football_player_position",
}


def prepare_knowledge_data():
    knowledge_data_dir = os.path.join(config.DATA_DIR, "knowledge")
    os.makedirs(knowledge_data_dir, exist_ok=True)
    output_file_path_template = os.path.join(knowledge_data_dir, "{relation_name}.json")

    counterfact_dataset_url = "https://rome.baulab.info/data/dsets/counterfact.json"

    counterfact_dataset = requests.get(counterfact_dataset_url, timeout=10).json()

    examples = [
        {
            "input": item["requested_rewrite"]["subject"],
            "output": item["requested_rewrite"]["target_true"]["str"],
            "relation_id": item["requested_rewrite"]["relation_id"],
            "relation_prompt": item["requested_rewrite"]["prompt"],
        }
        for item in counterfact_dataset
    ]

    examples_by_relation_id = {}
    for item in examples:
        if item["relation_id"] not in examples_by_relation_id:
            examples_by_relation_id[item["relation_id"]] = []
        examples_by_relation_id[item["relation_id"]].append(item)

    for relation_id, relation_name in RELATIONS_TO_INCLUDE.items():
        relation_examples = examples_by_relation_id[relation_id]
        relation_examples = {x["input"]: x["output"] for x in relation_examples}

        if relation_name == "location_religion":
            relation_examples = {k: v.replace("Christianity", "Christian") for k, v in relation_examples.items()}
            relation_examples = {k: v.replace("Judaism", "Jewish") for k, v in relation_examples.items()}
            relation_examples = {k: v.replace("Islam", "Muslim") for k, v in relation_examples.items()}
            # keep only Christian, Jewish, Muslim
            relation_examples = {k: v for k, v in relation_examples.items() if v in ["Christian", "Jewish", "Muslim"]}

        # write to file
        output_file_path = output_file_path_template.format(relation_name=relation_name)
        with open(output_file_path, "w") as f:
            json.dump(relation_examples, f, indent=4)


if __name__ == "__main__":
    prepare_knowledge_data()
