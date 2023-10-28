from typing import List, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer


def get_token_ids(tokenizer: PreTrainedTokenizer, token: str, all_capitalizations: bool = True):
    if all_capitalizations:
        tokens_to_check = [token.capitalize(), token.lower()]
    else:
        tokens_to_check = [token]

    token_ids = []
    for t in tokens_to_check:
        leading_space_tokens = tokenizer.encode(t, add_special_tokens=False)
        if len(leading_space_tokens) == 1:
            token_ids.append(leading_space_tokens[0])
        no_space_token = tokenizer.convert_tokens_to_ids(t)
        if no_space_token != 0:
            token_ids.append(no_space_token)

    return token_ids


def tokens_ranks(logits: torch.Tensor, tokenizer: PreTrainedTokenizer, tokens: List[str]):
    token_ids = {token: get_token_ids(tokenizer, token) for token in tokens}
    token_ranks = {token: token_ids_ranks(logits, token_ids[token]) for token in tokens}
    return token_ranks


def token_ids_ranks(logits: torch.Tensor, token_ids: torch.Tensor, keep_token_ids: torch.Tensor = None):
    vocab_size = len(logits)

    target_log_probs = logits[token_ids]

    all_log_probs = logits
    if keep_token_ids is not None:
        all_log_probs = logits[:, :, keep_token_ids]

    ranks = torch.sum(all_log_probs > target_log_probs.unsqueeze(-1), dim=-1) + 1

    return ranks.tolist()


def logits_top_tokens(logits: torch.Tensor, tokenizer: PreTrainedTokenizer, k: Optional[int] = None):
    logits = logits.cpu().float()

    # top_k_token_ids = torch.topk(logits, k=k, dim=-1).indices
    # Use argsort instead:
    top_token_ids = torch.argsort(logits, dim=-1, descending=True)
    if k is not None:
        # only keep top k along the last dimension
        top_token_ids = top_token_ids[..., :k]

    top_tokens = np.vectorize(tokenizer.decode)(top_token_ids)
    return top_tokens
