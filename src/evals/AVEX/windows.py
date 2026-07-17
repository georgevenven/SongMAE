import torch


def sliding_window_starts(valid_samples, context_samples):
    assert valid_samples > 0 and context_samples > 1
    if valid_samples <= context_samples:
        return [0]
    last = valid_samples - context_samples
    starts = list(range(0, last + 1, context_samples // 2))
    if starts[-1] != last:
        starts.append(last)
    return starts


def select_layers(selectors, num_layers):
    indices = []
    for selector in selectors:
        if selector == "all":
            values = range(num_layers)
        elif selector == "last_layer":
            values = [num_layers - 1]
        else:
            index = int(selector.removeprefix("layer_")) if isinstance(selector, str) else selector
            index += num_layers if index < 0 else 0
            assert 0 <= index < num_layers
            values = [index]
        indices.extend(index for index in values if index not in indices)
    assert indices
    return indices


def pool_embeddings(embeddings, valid_columns):
    assert embeddings.ndim == 4 and valid_columns.shape == embeddings.shape[:1]
    time = embeddings.shape[2]
    mask = torch.arange(time, device=embeddings.device)[None, :] < valid_columns[:, None]
    count = mask.sum()
    assert int(count) > 0
    return (embeddings * mask[:, None, :, None]).sum((0, 2)) / count
