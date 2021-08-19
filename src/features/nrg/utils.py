from src.features.nrg.constants import ATTR_TO_SPECIAL_TOKEN


def add_tokens_to_vocabulary(tokenizer, additional_tokens):
    num_added_norm_tokens = tokenizer.add_tokens(additional_tokens)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    return num_added_norm_tokens + num_added_tokens
