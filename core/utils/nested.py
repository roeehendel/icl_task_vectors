import collections.abc

import torch


def nested_apply(obj, func):
    if isinstance(obj, torch.Tensor):  # Check for torch.Tensor before checking for general iterables
        return func(obj)
    if isinstance(obj, collections.abc.Mapping):
        return {k: nested_apply(v, func) for k, v in obj.items()}
    elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes)):
        return type(obj)(nested_apply(item, func) for item in obj)
    else:
        return func(obj)


def is_mapping(obj):
    return hasattr(obj, "items")


def is_iterable(obj):
    return isinstance(obj, (list, tuple))


def nested_concat(obj_list):
    first_item = obj_list[0]

    if isinstance(first_item, torch.Tensor):
        return torch.cat(obj_list)
    elif is_mapping(first_item):
        keys = first_item.keys()
        return {key: nested_concat([d[key] for d in obj_list]) for key in keys}
    elif is_iterable(first_item):
        length = len(first_item)
        return type(first_item)([nested_concat([d[i] for d in obj_list]) for i in range(length)])
    else:
        raise ValueError(f"Unhandled data type: {type(first_item)}")
