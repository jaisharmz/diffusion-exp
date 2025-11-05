import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_batch, align_pair, get_zt
from generate import Tokenizer, generate
from model import NNModel

filename = "tinyshakespeare.txt"
df = open(filename, "r").read()[:1000]
unique_chars = set(df)
tokenizer = Tokenizer(unique_chars)
data_encoded = tokenizer(df)

model = NNModel(vocab_size=tokenizer.vocab_size, hidden_dim=64, num_layers=3, num_heads=16, 
                max_seq_len=256, bos_token_id=tokenizer.BOS_TOKEN, pad_token_id=tokenizer.PAD_TOKEN)

num_steps = 1000
batch_size = 32
max_seq_len = 128
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = []
for step in range(num_steps):
    x0_batch, x1_batch = get_batch(batch_size, max_seq_len, data_encoded, tokenizer)
    z1_total = []
    zt_total = []
    sub_mask_total = []
    ins_mask_total = []
    del_mask_total = []
    max_aligned_len = 0
    t_batch = torch.rand(batch_size)
    for i in range(batch_size):
        x0 = x0_batch[i].tolist()
        x1 = x1_batch[i].tolist()
        z0, z1 = align_pair(x0, x1, tokenizer)
        t_sample = t_batch[i].item()
        zt, edits, sub_mask, ins_mask, del_mask = get_zt(z0, z1, t_sample, tokenizer)
        z1_total.append(torch.tensor(z1, dtype=torch.long))
        zt_total.append(torch.tensor(zt, dtype=torch.long))
        sub_mask_total.append(sub_mask)
        ins_mask_total.append(ins_mask)
        del_mask_total.append(del_mask)

        if len(zt) > max_aligned_len:
            max_aligned_len = len(zt)
    for i in range(batch_size):
        pad_len = max_aligned_len - zt_total[i].shape[0]
        pad_tuple = (0, pad_len)
        z1_total[i] = F.pad(z1_total[i], pad_tuple, "constant", tokenizer.PAD_TOKEN)
        zt_total[i] = F.pad(zt_total[i], pad_tuple, "constant", tokenizer.PAD_TOKEN)
        sub_mask_total[i] = F.pad(sub_mask_total[i], pad_tuple, "constant", 0)
        ins_mask_total[i] = F.pad(ins_mask_total[i], pad_tuple, "constant", 0)
        del_mask_total[i] = F.pad(del_mask_total[i], pad_tuple, "constant", 0)
    z1_total = torch.stack(z1_total)
    zt_total = torch.stack(zt_total)
    sub_mask_total = torch.stack(sub_mask_total)
    ins_mask_total = torch.stack(ins_mask_total)
    del_mask_total = torch.stack(del_mask_total)
    padding_mask = (zt_total == tokenizer.PAD_TOKEN)
    loss = model.compute_loss(zt_total, t_batch, z1_total, sub_mask_total, ins_mask_total, del_mask_total, padding_mask)
    token_count = (~padding_mask).sum()
    loss = torch.sum(loss) / token_count
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
            padding_mask = zt == tokenizer.PAD_TOKEN
            zt = generate(model, zt, t, padding_mask)
            strings = tokenizer.decode(zt)
            print(strings)
            print(zt)