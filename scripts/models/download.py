from dotenv import load_dotenv
from core.experiments_config import MODELS_TO_EVALUATE

load_dotenv(".env")

from core.models.llm_loading import load_model_and_tokenizer


def main():
    for model_params in MODELS_TO_EVALUATE:
        print("Downloading", model_params)
        load_model_and_tokenizer(*model_params, load_to_cpu=True)


if __name__ == "__main__":
    main()
