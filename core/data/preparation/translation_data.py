from dotenv import load_dotenv

from core import config

load_dotenv(".env")

import os
import json
import requests

import nltk
from nltk.corpus import wordnet as wn


def prepare_translation_data():
    translation_data_dir = os.path.join(config.DATA_DIR, "translation")
    os.makedirs(translation_data_dir, exist_ok=True)
    output_file_path_template = os.path.join(translation_data_dir, "{mapping_name}.json")

    common_words_url_template = "https://raw.githubusercontent.com/frekwencja/most-common-words-multilingual/main/data/wordfrequency.info/{language}.txt"

    BASE_LANGUAGE = "en"
    LANGUAGES = ["es", "it", "fr"]

    # download wordnet
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    # Here, we instead load to a dictionary
    most_common_words = {}
    for language in LANGUAGES + [BASE_LANGUAGE]:
        url = common_words_url_template.format(language=language)
        r = requests.get(url, timeout=5)
        most_common_words[language] = r.text.splitlines()

    for language in LANGUAGES:
        base_language_words = most_common_words[BASE_LANGUAGE]
        other_language_words = most_common_words[language]

        # keep only first n words
        num_words = 1000
        base_language_words = base_language_words[:num_words]
        other_language_words = other_language_words[:num_words]

        # to lowercase
        base_language_words = [x.lower() for x in base_language_words]
        other_language_words = [x.lower() for x in other_language_words]

        mapping = {
            base_word.strip(): other_word.strip()
            for base_word, other_word in zip(base_language_words, other_language_words)
        }
        inverse_mapping = {v: k for k, v in mapping.items()}

        # filter out words that are not in wordnet (in the values)
        lang_to_wn_lang = {"it": "ita", "fr": "fra", "es": "spa", "en": "eng"}
        wn_base_language = lang_to_wn_lang[BASE_LANGUAGE]
        wn_language = lang_to_wn_lang[language]
        mapping = {k: v for k, v in mapping.items() if len(wn.synsets(v, lang=wn_language)) > 0}
        inverse_mapping = {k: v for k, v in inverse_mapping.items() if len(wn.synsets(v, lang=wn_base_language)) > 0}

        with open(output_file_path_template.format(mapping_name=f"{BASE_LANGUAGE}_{language}"), "w") as f:
            json.dump(mapping, f, indent=4)

        with open(output_file_path_template.format(mapping_name=f"{language}_{BASE_LANGUAGE}"), "w") as f:
            json.dump(inverse_mapping, f, indent=4)


if __name__ == "__main__":
    prepare_translation_data()
