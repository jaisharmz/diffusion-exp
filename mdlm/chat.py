import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import os
from model import get_model_and_tokenizer
from generator import Generator
from chat_utils import single_turn_generate
from scheduler import LinearAlphaScheduler

model, tokenizer = get_model_and_tokenizer("/home/jaisharma/diffusion-exp/mdlm/models/ModernBERT-large/alpaca/checkpoint-final")
scheduler = LinearAlphaScheduler()
generator = Generator(model, tokenizer, scheduler)
single_turn_generate(generator)