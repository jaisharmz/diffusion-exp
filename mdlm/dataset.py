import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import functools

def tokenize_and_group(examples, tokenizer, text_field, seq_length):
    tokenized = tokenizer(examples[text_field])
    ids = tokenized["input_ids"]
    concatenated = list(chain.from_iterable(ids))
    if not concatenated:
        return {"input_ids": [], "labels": []}
    total_length = len(concatenated) // seq_length * seq_length
    concatenated = concatenated[:total_length]
    chunks = [concatenated[i:i+seq_length] for i in range(0, total_length)]
    return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

def get_tokenize_function(tokenizer, text_field, seq_length):
    tokenize_func = functools.partial(
        tokenize_and_group, 
        tokenizer=tokenizer, 
        text_field=text_field, 
        seq_length=seq_length
    )
    return tokenize_func