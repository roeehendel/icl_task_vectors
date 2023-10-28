import os
import json
import requests

from core import config


def most_common_value(lst):
    # return most common value
    return max(set(lst), key=lst.count)


def prepare_verb_conjugations():
    linguistic_data_dir = os.path.join(config.DATA_DIR, "linguistic")
    os.makedirs(linguistic_data_dir, exist_ok=True)
    output_file_path_template = os.path.join(linguistic_data_dir, "{mapping_name}.json")

    verbs_conjugations_url = (
        "https://raw.githubusercontent.com/Drulac/English-Verbs-Conjugates/master/verbs-conjugations.json"
    )

    # load verbs conjugations
    verbs_conjugations = requests.get(verbs_conjugations_url, timeout=5).json()

    mappings = {}

    # 1. regular verb to past simple (imperfect) [e.g. "go" -> "went"]
    mappings["present_simple_past_simple"] = {
        verb["verb"]: most_common_value(verb["indicative"]["imperfect"])
        for verb in verbs_conjugations
        if "imperfect" in verb["indicative"] and len(verb["indicative"]["imperfect"]) > 0
    }
    # 2. regular verb to gerund [e.g. "go" -> "going"]
    mappings["present_simple_gerund"] = {
        verb["verb"]: most_common_value(verb["gerund"])
        for verb in verbs_conjugations
        if "gerund" in verb and len(verb["gerund"]) > 0
    }
    # 3. regular verb to past perfect [e.g. "go" -> "gone"]
    mappings["present_simple_past_perfect"] = {
        verb["verb"]: most_common_value(verb["indicative"]["perfect"])
        for verb in verbs_conjugations
        if "perfect" in verb["indicative"] and len(verb["indicative"]["perfect"]) > 0
    }
    # filter out past_perfect that end with "ed"
    mappings["present_simple_past_perfect"] = {
        k: v for k, v in mappings["present_simple_past_perfect"].items() if not v.endswith("ed")
    }
    # keep only alphabet letters
    mappings = {k: {k2: v2 for k2, v2 in v.items() if k2.isalpha() and v2.isalpha()} for k, v in mappings.items()}

    # Save to files
    os.makedirs(linguistic_data_dir, exist_ok=True)
    for mapping_name, mapping in mappings.items():
        output_file_path = output_file_path_template.format(mapping_name=mapping_name)
        with open(output_file_path, "w") as f:
            json.dump(mapping, f, indent=4)


def prepare_singular_plural():
    linguistic_data_dir = os.path.join(config.DATA_DIR, "linguistic")
    # output_file_path = os.path.join(linguistic_data_dir, "singular_plural.json")
    output_file_path = os.path.join(linguistic_data_dir, "plural_singular.json")

    singular_plural_url = "https://raw.githubusercontent.com/sindresorhus/irregular-plurals/main/irregular-plurals.json"

    singular_plural = requests.get(singular_plural_url, timeout=5).json()
    plural_singular = {v: k for k, v in singular_plural.items()}

    # save to file
    with open(output_file_path, "w") as f:
        json.dump(plural_singular, f, indent=4)


def prepare_linguistic_data():
    prepare_verb_conjugations()
    prepare_singular_plural()


if __name__ == "__main__":
    prepare_linguistic_data()
