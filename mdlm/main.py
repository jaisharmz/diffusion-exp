import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from dataset import get_tokenize_function
from model import get_model_and_tokenizer
from trainer import Trainer, TrainingArguments

model, tokenizer = get_model_and_tokenizer("google-bert/bert-base-uncased")
dataset = load_dataset("Trelis/tiny-shakespeare")
tokenize_function = get_tokenize_function(tokenizer, "Text", 128)
dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
training_args = TrainingArguments()
trainer = Trainer(model, tokenizer, 
                  train_dataset=dataset["train"],
                  eval_dataset=dataset["test"],
                  args=training_args,
                  data_collator=transformers.DataCollatorForSeq2Seq(
                        tokenizer,
                        return_tensors="pt",
                        padding=True,
                    ))
trainer.train()