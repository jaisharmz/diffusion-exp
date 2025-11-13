import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import functools
import transformers

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

def default_sft_map_fn(row, tokenizer, mask_prompt_loss=True):
    prompt_response_tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
    labels = prompt_response_tokens.copy()
    if mask_prompt_loss:
        prompt_tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True)
        labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)
        return {
            "prompt_response_tokens": prompt_response_tokens,
            "labels": labels,
            "prompt_len": len(prompt_tokens)
        }
    return {"input_ids": prompt_response_tokens, "labels": labels}

def get_map_function(tokenizer, mask_prompt_loss=True):
    map_function = functools.partial(
        default_sft_map_fn, 
        tokenizer=tokenizer, 
        mask_prompt_loss=mask_prompt_loss
    )
    return map_function

class NoAttentionMaskCollator(transformers.DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        outputs = super().__call__(features, return_tensors)
        outputs.pop("attention_mask")
        return outputs