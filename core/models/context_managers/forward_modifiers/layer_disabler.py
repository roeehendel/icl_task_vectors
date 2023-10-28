from typing import Iterable

from torch import nn
from transformers import PreTrainedModel
from core.models.utils.llm_layers import get_layers, get_layers_path, set_nested_attr


class LayerDisabler:
    def __init__(self, model: PreTrainedModel, layers_to_disable: Iterable[int] = ()):
        self._model = model
        self._layers_to_disable = layers_to_disable
        self._layers_path = get_layers_path(model)
        self._original_layers = get_layers(model)
        self._hooks = []

    def __enter__(self):
        new_layers = nn.ModuleList(
            [layer for i, layer in enumerate(self._original_layers) if i not in self._layers_to_disable]
        )
        set_nested_attr(self._model, self._layers_path, new_layers)

    def __exit__(self, exc_type, exc_value, traceback):
        set_nested_attr(self._model, self._layers_path, self._original_layers)
