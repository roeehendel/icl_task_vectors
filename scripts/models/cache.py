from dotenv import load_dotenv

from core.experiments_config import MODELS_TO_EVALUATE

load_dotenv(".env")

import os
import subprocess
import time

import schedule
import torch
from tqdm.auto import tqdm

from core.models.llm_loading import get_local_path


def cache_directory(directory: str):
    # run cat file > /dev/null for each file in directory

    # recursively list all files in directory
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_paths.append(os.path.join(root, filename))

    pbar = tqdm(file_paths, leave=False)

    for filename in pbar:
        pbar.set_description(f"Caching {filename}")
        subprocess.run(["cat", os.path.join(directory, filename)], stdout=subprocess.DEVNULL, check=True)


def cache_model():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ": Loading models")

    success = []
    error = []

    for model_params in MODELS_TO_EVALUATE:
        try:
            model_dir = get_local_path(*model_params)
            # first check if directory exists
            if not os.path.exists(model_dir):
                continue
                # # download the model
                # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ": Downloading", model_params)
                # load_model(*model_params, load_to_cpu=True)

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ": Caching", model_params)
            cache_directory(model_dir)

            success.append(model_params)

        except Exception as e:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ": Error caching", model_params)
            print(e)
            error.append(model_params)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ": Models loaded")
    print(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ": Results summary.", "Success:", success, "Error:", error
    )

    torch.cuda.empty_cache()


def main():
    # make no devices visible
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # schedule.every().day.at("08:00").do(load_model)
    schedule.every(5).minutes.do(cache_model)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ": Starting model cache")

    cache_model()
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
    # load_model()
