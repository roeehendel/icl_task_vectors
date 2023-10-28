import nltk
from transformers import PreTrainedTokenizer

from core import config
from core.data.tasks.mapping_task import MappingTask

nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import wordnet as wn
from typing import Any


class TranslationTask(MappingTask):
    @staticmethod
    def _get_synonyms(word: str, lang_to: str):
        lang = {
            "en": "eng",
            "fr": "fra",
            "it": "ita",
            "es": "spa",
        }[lang_to]
        synonyms = [word]
        for syn in wn.synsets(word, lang=lang):
            for lemma in syn.lemmas(lang=lang):
                synonyms.append(lemma.name())
        return synonyms

    def compare_outputs(self, output1: Any, output2: Any) -> bool:
        output1, output2 = output1.strip(), output2.strip()
        output_lang = self.mapping_name.split("_")[1]
        synonyms1 = self._get_synonyms(output1, output_lang)
        synonyms2 = self._get_synonyms(output2, output_lang)
        return len(set(synonyms1) & set(synonyms2)) > 0
