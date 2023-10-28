import torch
from transformers import PreTrainedModel

from core.models.utils.llm_layers import get_layers


class HiddenInjector:
    def __init__(
        self,
        model: PreTrainedModel,
        injection_layers: torch.Tensor,  # (batch_size)
        injection_positions: torch.Tensor,  # (batch_size)
        hiddens_to_inject: torch.Tensor,  # (batch_size, hidden_size)
    ):
        """
        Args:
            model: The model to inject hidden states into
            injection_layer: the layer to inject hidden states into, for each example in the batch (batch_size)
            injection_position: the position to inject hidden states into, for each example in the batch (batch_size)
            hidden_to_inject: the hidden states to inject, for each example in the batch (batch_size, hidden_size)
        """

        self._model = model
        self._injection_layer = injection_layers
        self._injection_position = injection_positions
        self._hidden_to_inject = hiddens_to_inject

        self._hooks = []

    def __enter__(self):
        self._register_forward_hooks()

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()

    def _register_forward_hooks(self):
        def inject_hidden_hook(layer_idx):
            def inject_hidden(mod, inp, out):
                hidden_states = out[0] if isinstance(out, tuple) else out

                mask = self._injection_layer == layer_idx
                if mask.any():
                    hidden_to_inject = self._hidden_to_inject.to(hidden_states.device).type(hidden_states.dtype)
                    idx_to_inject = torch.arange(hidden_states.shape[0], device=hidden_states.device)[mask]
                    hidden_states[idx_to_inject, self._injection_position[mask]] = hidden_to_inject[mask]

                return out

            return inject_hidden

        for i, layer in enumerate(get_layers(self._model)):
            hook = layer.register_forward_hook(inject_hidden_hook(i))
            self._hooks.append(hook)
