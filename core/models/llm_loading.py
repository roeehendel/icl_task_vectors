import math
import os
from typing import Literal, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from accelerate import init_empty_weights
from accelerate.utils.modeling import infer_auto_device_map, get_balanced_memory

from core.models.utils.llm_layers import get_layers, get_layers_path

BASE_KWARGS = {
    "torch_dtype": torch.float16,
    "trust_remote_code": True,
}

GPU_KWARGS = {
    **BASE_KWARGS,
    # "load_in_8bit": True,
    # "device_map": "auto",
}

CPU_KWARGS = {
    **BASE_KWARGS,
}

LlamaVariant = Literal["huggingface", "vicuna"]
LlamaSize = Literal["7B", "13B"]


def _setup_tokenizer(tokenizer: PreTrainedTokenizer) -> None:
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token


def llama_local_path(variant: LlamaVariant, size: LlamaSize) -> str:
    llama_dir = os.environ["LLAMA_DIR"]
    return f"{llama_dir}/{variant}/{size}"


def get_local_path(model_type: str, model_variant: str) -> str:
    if model_type == "llama":
        return llama_local_path("huggingface", model_variant)

    model_path = get_model_path(model_type, model_variant)
    username, model_name = model_path.split("/")

    huggingface_cache_dir = os.environ["TRANSFORMERS_CACHE"]
    return f"{huggingface_cache_dir}/models--{username}--{model_name}"


def get_model_path(model_type: str, model_variant: str) -> str:
    model_path = MODEL_PATHS[model_type][model_variant]
    return model_path


def _get_falcon_device_map() -> dict[str, int]:
    num_devices = torch.cuda.device_count()
    device_map = {
        "transformer.word_embeddings": 0,
        "lm_head": 0,
        "transformer.ln_f": 0,
    }
    num_layers = 60
    num_layers_per_device = math.ceil(num_layers / (num_devices - 1))
    device_map.update({f"transformer.h.{i}": (i // num_layers_per_device + 1) for i in range(num_layers)})
    return device_map


def _create_device_map(model_path: str) -> dict[str, int]:
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    layer_class = get_layers(model)[0].__class__.__name__

    max_memory = get_balanced_memory(model, no_split_module_classes=[layer_class])
    max_memory[0] = 0
    base_device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=[layer_class])

    num_devices = torch.cuda.device_count()

    layers_path = get_layers_path(model)

    # device_map_lm_head = {k: v for k, v in base_device_map.items() if "lm_head" in k}
    # device_map_emb = {k: v for k, v in base_device_map.items() if "emb" in k}
    device_map_layers = {k: v for k, v in base_device_map.items() if k.startswith(layers_path)}
    device_map_other = {k: v for k, v in base_device_map.items() if k not in device_map_layers}

    # place the other layers on device 0
    device_map_other = {k: 0 for k in device_map_other}
    # split the layers evenly across the other devices (1-num_devices)
    num_layers = len(device_map_layers)
    num_layers_per_device = math.ceil(num_layers / (num_devices - 1))
    device_map_layers = {k: (i // num_layers_per_device + 1) for i, k in enumerate(device_map_layers)}

    device_map = {**device_map_other, **device_map_layers}

    return device_map


def load_model(model_type: str, model_variant: str, load_to_cpu: bool = False):
    model_path = get_model_path(model_type, model_variant)

    kwargs = CPU_KWARGS if load_to_cpu else GPU_KWARGS

    kwargs["device_map"] = _create_device_map(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model = model.eval()  # check if this is necessary

    return model


def load_tokenizer(model_type: str, model_variant: str) -> PreTrainedTokenizer:
    model_path = get_model_path(model_type, model_variant)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    _setup_tokenizer(tokenizer)

    return tokenizer


def load_model_and_tokenizer(
    model_type: str, model_variant: str, load_to_cpu: bool = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(model_type, model_variant)
    model = load_model(model_type, model_variant, load_to_cpu=load_to_cpu)

    return model, tokenizer


MODEL_PATHS = {
    "pythia": {
        "1.4B": "EleutherAI/pythia-1.4b",
        "2.8B": "EleutherAI/pythia-2.8b",
        "6.9B": "EleutherAI/pythia-6.9b",
        "12B": "EleutherAI/pythia-12b",
    },
    "llama": {
        "7B": llama_local_path("huggingface", "7B"),
        "13B": llama_local_path("huggingface", "13B"),
        "30B": llama_local_path("huggingface", "30B"),
        "65B": llama_local_path("huggingface", "65B"),
    },
    "falcon": {
        "7B": "tiiuae/falcon-7b",
        "40B": "tiiuae/falcon-40b",
    },
    "gpt-j": {
        "6B": "EleutherAI/gpt-j-6B",
    },
    "gpt-2": {
        "0.35B": "gpt2-medium",
        "0.77B": "gpt2-large",
        "1.5B": "gpt2-xl",
    },
    "mpt": {
        "7B": "mosaicml/mpt-7b",
    },
    "gpt-neox": {
        "20B": "EleutherAI/gpt-neox-20b",
    },
    "starcoder": {
        "regular": "bigcode/starcoder",
        "plus": "bigcode/starcoderplus",
    },
    "cerebras-gpt": {
        "6.7B": "cerebras/Cerebras-GPT-6.7B",
        "13B": "cerebras/Cerebras-GPT-13B",
    },
}
