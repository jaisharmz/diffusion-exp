import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_batch, align_pair, get_zt, Edit
from generate import Tokenizer, generate
from model import NNModel

filename = "tinyshakespeare.txt"
df = open(filename, "r").read()[:1000]
unique_chars = set(df)
tokenizer = Tokenizer(unique_chars)
data_encoded = tokenizer(df)
device = "cpu"

model = NNModel(vocab_size=tokenizer.vocab_size, hidden_dim=64, num_layers=3, num_heads=16, 
                max_seq_len=256, bos_token_id=tokenizer.BOS_TOKEN, pad_token_id=tokenizer.PAD_TOKEN,
                tokenizer=tokenizer, device=device)
model.to(device)

num_steps = 1000
batch_size = 32
max_seq_len = 128
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = []
for step in range(num_steps):
    x0_batch, x1_batch = get_batch(batch_size, max_seq_len, data_encoded, tokenizer)
    inputs = {"x0_ids": x0_batch, "x1_ids": x1_batch}
    loss, out = model.compute_loss(inputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step % 10 == 0:
        print(f"Step {step}: Loss {loss.item():.4f}")
    if step % 100 == 0 or step == num_steps - 1:
        strings = ["hello world!", "wow this is cool", ""]
        zt = tokenizer(strings)
        for i in range(10):
            zt = tokenizer.to_tensor(zt)
            batch_size, seq_len = zt.shape
            t = torch.rand(batch_size)
            attention_mask = zt != tokenizer.PAD_TOKEN
            zt = generate(model, zt, attention_mask, t)
            strings = tokenizer.decode(zt)
            print(strings)
            print(zt)