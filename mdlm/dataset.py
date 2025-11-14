import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import functools
import transformers
from datasets import load_dataset, concatenate_datasets, DatasetDict

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
    input_ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
    labels = input_ids.copy()
    if mask_prompt_loss:
        prompt_tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True)
        labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)
    return {"input_ids": input_ids, "labels": labels}

def get_map_function(model, tokenizer, mask_prompt_loss=True):
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
    
def load_sft_dataset(dataset_args):
    specs = [p.strip() for p in dataset_args.split("|") if p.strip()]
    train_datasets = []
    test_datasets = []
    for spec in specs:
        print(f"Loading {spec}...")
        if "tulu-3" in spec:
            ds = load_dataset("allenai/tulu-3-sft-mixture")
            ds = ds["train"].train_test_split(test_size=0.05, seed=0)
        elif "smoltalk" in spec:
            ds = load_dataset("HuggingFaceTB/smoltalk", "all")
            if "test" not in ds:
                ds = ds["train"].train_test_split(test_size=0.05, seed=0)
        else:
            ds = load_dataset(spec)
            if "test" not in ds:
                ds = ds["train"].train_test_split(test_size=0.05, seed=0)
        ds = ds.select_columns(["messages"])
        train_datasets.append(ds["train"])
        test_datasets.append(ds["test"])
    print(f"Merging {len(train_datasets)} datasets...")
    combined_dataset = DatasetDict({
        "train": concatenate_datasets(train_datasets),
        "test": concatenate_datasets(test_datasets)
    })
    combined_dataset["train"] = combined_dataset["train"].shuffle(seed=0).select(range(100000))
    combined_dataset["test"] = combined_dataset["test"].shuffle(seed=0).select(range(10000))
    return combined_dataset

def group_texts(examples, max_length):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def get_group_texts_function(max_length):
    group_texts_function = functools.partial(
        group_texts, 
        max_length=max_length
    )
    return group_texts_function