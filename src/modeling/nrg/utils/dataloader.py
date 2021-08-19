from collections import defaultdict
from itertools import chain
import torch

GPT2_DOUBLE_HEADS_MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
GPT2_DOUBLE_HEADS_PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


def pad_double_heads_inputs(batch_inputs, pad_token):
    max_l = max(len(x) for x in batch_inputs["input_ids"])
    for name in GPT2_DOUBLE_HEADS_PADDED_INPUTS:
        batch_inputs[name] = [x + [pad_token if name != "lm_labels" else -100] * (max_l - len(x)) for x in batch_inputs[name]]
    return batch_inputs

def collate_double_heads_data(batch, pad_token):
    num_candidates = len(batch[0])
    batch_inputs = defaultdict(list)
    chained_batch = chain.from_iterable(batch)

    for instance in chained_batch:
        for field, data in instance.items():
            batch_inputs[field].append(data)

    padded_inputs = pad_double_heads_inputs(batch_inputs, pad_token)

    input_tensors = []
    batch_size = tuple([len(batch_inputs[GPT2_DOUBLE_HEADS_MODEL_INPUTS[0]])//num_candidates])

    for input_name in GPT2_DOUBLE_HEADS_MODEL_INPUTS:
        tensor = torch.tensor(padded_inputs[input_name])

        if input_name != "mc_labels":
            tensor = tensor.view((-1, num_candidates) + tensor.shape[1:])
        else:
            tensor = torch.ones(size=batch_size, dtype=torch.long) * (num_candidates - 1)
        input_tensors.append(tensor)

    return input_tensors