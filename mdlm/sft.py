import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import os
from datasets import load_dataset, concatenate_datasets
from dataset import get_map_function, NoAttentionMaskCollator, load_sft_dataset, get_group_texts_function
from model import get_model_and_tokenizer
from trainer import Trainer, TrainingArguments

max_length = 512
model, tokenizer = get_model_and_tokenizer("answerdotai/ModernBERT-base")
dataset = load_sft_dataset("allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk")
map_function = get_map_function(model, tokenizer, True)
group_texts_function = get_group_texts_function(max_length)
dataset = dataset.map(map_function)
dataset = dataset.map(group_texts_function, batched=True, batch_size=1000, num_proc=4)
dataset = dataset.filter(lambda row: len(row["input_ids"]) <= max_length)
training_args = TrainingArguments(output_dir="models/ModernBERT-large/alpaca", logging_steps=1, eval_steps=50)
trainer = Trainer(model=model, tokenizer=tokenizer, 
                  train_dataset=dataset["train"],
                  eval_dataset=dataset.get("test", None),
                  args=training_args,
                  data_collator=NoAttentionMaskCollator(
                      tokenizer,
                      return_tensors="pt",
                      padding=True,
                      label_pad_token_id=-100
                  ))
trainer.train()
trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
trainer.processing_class.save_pretrained(os.path.join(training_args.output_dir, "checkpoint-final"))