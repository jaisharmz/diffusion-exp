# === main3.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from model import NNModel  # use the fixed model.py

# ------------------------
# Simple Tokenizer
# ------------------------
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<BOS>": 1, "<GAP>": 2}
        self.inv_vocab = {0: "<PAD>", 1: "<BOS>", 2: "<GAP>"}
        self.PAD_TOKEN = 0
        self.BOS_TOKEN = 1
        self.GAP_TOKEN = 2

    def encode(self, text):
        for ch in text:
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.inv_vocab[idx] = ch
        return [self.vocab[ch] for ch in text]

    def decode(self, tokens):
        return "".join([self.inv_vocab.get(t, "") for t in tokens if t not in (self.PAD_TOKEN, self.GAP_TOKEN)])

    def to_tensor(self, batch):
        max_len = max(len(seq) for seq in batch)
        tensor = torch.full((len(batch), max_len), self.PAD_TOKEN, dtype=torch.long)
        for i, seq in enumerate(batch):
            tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return tensor


# ------------------------
# Data batching utilities
# ------------------------
def get_batch(batch_size, seq_len, data_encoded, tokenizer):
    # FIX: make x0 and x1 overlapping sequences instead of unrelated randoms
    x0 = []
    x1 = []
    for _ in range(batch_size):
        i = random.randint(0, len(data_encoded) - seq_len - 5)
        shift = random.randint(1, 4)
        x0_chunk = data_encoded[i : i + seq_len]
        x1_chunk = data_encoded[i + shift : i + seq_len + shift]
        x0.append([tokenizer.BOS_TOKEN] + x0_chunk)
        x1.append([tokenizer.BOS_TOKEN] + x1_chunk)
    x0 = tokenizer.to_tensor(x0)
    x1 = tokenizer.to_tensor(x1)
    return x0, x1


# ------------------------
# Alignment utilities
# ------------------------
def align_pair(z0, z1, pad_token, gap_token):
    """
    Simple pair alignment: fill gaps with <GAP> tokens so both have same length.
    """
    len0, len1 = len(z0), len(z1)
    max_len = max(len0, len1)
    z0_aligned = z0 + [gap_token] * (max_len - len0)
    z1_aligned = z1 + [gap_token] * (max_len - len1)
    return torch.tensor(z0_aligned), torch.tensor(z1_aligned)


def get_zt(z0, z1, t, pad_token, gap_token):
    """
    Linear interpolation between z0 and z1 using time t (like noise schedule).
    """
    zt = []
    for a, b in zip(z0, z1):
        if random.random() < t:
            zt.append(b)
        else:
            zt.append(a)
    return torch.tensor(zt)


# ------------------------
# Generation
# ------------------------
@torch.no_grad()
def generate(model, tokenizer, seq_len=100, steps=50, tau=0.2):  # FIX: higher tau
    zt = torch.full((1, seq_len), tokenizer.PAD_TOKEN, dtype=torch.long, device="cuda")
    for step in range(steps):
        t = torch.tensor([1 - step / steps]).to("cuda")
        padding_mask = zt == tokenizer.PAD_TOKEN
        sub_logits, ins_logits, del_logits = model(zt, t, padding_mask)
        sub_probs = F.softmax(sub_logits / tau, dim=-1)
        del_probs = torch.sigmoid(del_logits)

        zt_next = zt.clone()
        for i in range(seq_len):
            if random.random() < del_probs[0, i].item():
                zt_next[0, i] = tokenizer.PAD_TOKEN
            elif random.random() < 0.3:
                zt_next[0, i] = torch.multinomial(sub_probs[0, i], 1)
        zt = zt_next

    return tokenizer.decode(zt[0].tolist())


# ------------------------
# Training Loop
# ------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = SimpleTokenizer()
    data = open("tinyshakespeare.txt").read()
    data_encoded = tokenizer.encode(data)
    pad_token_id = tokenizer.PAD_TOKEN

    model = NNModel(len(tokenizer.vocab), pad_token_id=pad_token_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(1000):
        x0, x1 = get_batch(8, 64, data_encoded, tokenizer)
        x0, x1 = x0.to(device), x1.to(device)

        aligned = [align_pair(a, b, pad_token_id, tokenizer.GAP_TOKEN) for a, b in zip(x0.tolist(), x1.tolist())]
        z0 = torch.stack([a[0] for a in aligned]).to(device)
        z1 = torch.stack([a[1] for a in aligned]).to(device)

        padding_mask = z1 == pad_token_id
        t_batch = torch.rand(x0.size(0), device=device)

        # Construct z_t by interpolating between z0 and z1
        zt = torch.stack([
            get_zt(z0[i], z1[i], t_batch[i].item(), pad_token_id, tokenizer.GAP_TOKEN)
            for i in range(x0.size(0))
        ]).to(device)

        # Compute and optimize loss
        loss = model.compute_loss(zt, z1, padding_mask, t_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")

    # Optional: try sampling
    print("Generated text:")
    print(generate(model, tokenizer))
