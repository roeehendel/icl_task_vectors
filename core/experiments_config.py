TASKS_TO_EVALUATE = [
    # One for each category
    "translation_fr_en",
    "linguistic_present_simple_gerund",
    "knowledge_country_capital",
    "algorithmic_next_letter",
    # Translation
    "translation_es_en",
    "translation_en_fr",
    "translation_en_es",
    # Linguistic
    "linguistic_present_simple_past_simple",
    "linguistic_plural_singular",
    "linguistic_antonyms",
    # Knowledge
    "knowledge_person_language",
    "knowledge_location_continent",
    "knowledge_location_religion",
    # Algorithmic
    "algorithmic_prev_letter",
    "algorithmic_list_first",
    "algorithmic_list_last",
    "algorithmic_to_upper",
    "algorithmic_to_lower",
]

MODELS_TO_EVALUATE = [
    # ("gpt-2", "1.5B"),
    ("pythia", "2.8B"),
    ("llama", "7B"),
    ("gpt-j", "6B"),
    ("pythia", "6.9B"),
    ("llama", "13B"),
    ("pythia", "12B"),
    ("llama", "30B"),
    # ("mpt", "7B"), # error in ForwardTracer
    # ("falcon", "7B"), # error in past_key_values
]
