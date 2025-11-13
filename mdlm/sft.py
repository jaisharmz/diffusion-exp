import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset, concatenate_datasets
from dataset import get_map_function, NoAttentionMaskCollator
from model import get_model_and_tokenizer
from trainer import Trainer, TrainingArguments

max_length = 512
model, tokenizer = get_model_and_tokenizer("answerdotai/ModernBERT-base")
ds_tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train")
ds_smol = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
ds_tulu = ds_tulu.select_columns(["messages"])
ds_smol = ds_smol.select_columns(["messages"])
dataset = concatenate_datasets([ds_tulu, ds_smol])
map_function = get_map_function(tokenizer, True)
dataset = dataset.map(map_function)
dataset = dataset.filter(lambda row: len(row["input_ids"]) <= max_length)
training_args = TrainingArguments(output_dir="models/ModernBERT-large/alpaca")
trainer = Trainer(model=model, tokenizer=tokenizer, 
                  train_dataset=dataset["train"],
                  eval_dataset=dataset.get("test", None),
                  args=training_args,
                  data_collator=NoAttentionMaskCollator(
                      tokenizer,
                      return_tensors="pt",
                      padding=True,
                      label_pad_token_id=tokenizer.pad_token_id
                  ))
trainer.train()
