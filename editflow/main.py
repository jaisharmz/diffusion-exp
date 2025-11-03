# Load data
# Make x0, x1
# Get z0, z1
# Align using DP
# Get zt
# Get remaining edits
# define model (compute loss)
# model training loop

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

filename = "/mnt/c/Users/jais/Documents/diffusion-exp/editflow/tinyshakespeare.txt"
df = open(filename, "r").read()
unique_chars = set(df)
id_to_char = {i : char for i, char in enumerate(unique_chars)}
char_to_id = {char : i for i, char in enumerate(unique_chars)}
data_encoded = [char_to_id[char] for char in df]
vocab_size = len(unique_chars)
PAD_TOKEN = vocab_size
GAP_TOKEN = vocab_size + 1
vocab_size = vocab_size + 2
id_to_char[PAD_TOKEN] = "<PAD>"
id_to_char[GAP_TOKEN] = "<GAP>"

batch_size = 64
seq_len = 128

def get_batch(batch_size, seq_len):
    x0 = []
    x1 = []
    for batch in range(batch_size):
        idx = random.randint(0,len(data_encoded) - seq_len)
        random_chunk = data_encoded[idx:idx + seq_len]
        x0.append(random_chunk)
        idx = random.randint(0,len(data_encoded) - seq_len)
        random_chunk = data_encoded[idx:idx + seq_len]
        x1.append(random_chunk)
    x0 = torch.tensor(x0)
    x1 = torch.tensor(x1)
    return x0, x1

def align_pair(x0, x1):
    dp = [[0 for j in range(len(x1) + 1)] for i in range(len(x0) + 1)]
    for i in range(len(dp)):
        dp[i][0] = i
    for j in range(len(dp[0])):
        dp[0][j] = j
    for i in range(1,len(dp)):
        for j in range(1,len(dp[0])):
            cost = 0
            if x0[i-1] != x1[j-1]:
                cost = 1
            dp[i][j] = min(1 + dp[i][j-1], 1 + dp[i-1][j], cost + dp[i-1][j-1])
    z0 = []
    z1 = []
    i = len(dp) - 1
    j = len(dp[0]) - 1
    while i > 0 or j > 0:
        cost = 0
        if i > 0 and j > 0 and x0[i-1] != x1[j-1]:
            cost = 1
        if i > 0 and j > 0 and dp[i][j] == cost + dp[i-1][j-1]:
            z0.append(x0[i-1])
            z1.append(x1[j-1])
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == 1 + dp[i][j-1]:
            z0.append(GAP_TOKEN)
            j -= 1
        else:
            z1.append(GAP_TOKEN)
            i -= 1
    z0 = z0 + [GAP_TOKEN] * i
    z1 = z1 + [GAP_TOKEN] * j
    z0.reverse()
    z1.reverse()
    return z0, z1

def get_zt(z0, z1):
    # 0 = substitute, 1 = add, 2 = delete
    edit_indices = [i for i in range(len(z0)) if z0[i] != z1[i]]
    k = random.randint(0, len(edit_indices))
    indices_to_edit = random.sample(edit_indices, k)
    zt = z0[:]
    edits = []
    sub_mask = [0 for i in range(len(z0))]
    ins_mask = [0 for i in range(len(z0))]
    del_mask = [0 for i in range(len(z0))]
    for i in indices_to_edit:
        zt[i] = z1[i]
    for i in set(edit_indices) - set(indices_to_edit):
        edit = 0
        if z1[i] == GAP_TOKEN:
            edit = 2
            del_mask[i] = 1
        elif z0[i] == GAP_TOKEN:
            edit = 1
            ins_mask[i] = 1
        else:
            sub_mask[i] = 1
        edits.append([i, edit])

    sub_mask = torch.tensor(sub_mask)
    ins_mask = torch.tensor(ins_mask)
    del_mask = torch.tensor(del_mask)
    return zt, edits, sub_mask, ins_mask, del_mask

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        t = t.view(-1, 1)
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class NNModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, max_seq_len, bos_token_id, pad_token_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim))
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
                                       dropout=0.1, activation="gelu", batch_first=False)
            for i in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.rates_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.prob_ins = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.prob_sub = self.rates_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.mse = nn.MSELoss(reduce="none")
        self.ce = nn.CrossEntropyLoss(reduce="none")
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
    def forward(self, zt, t, padding_mask):
        # predict lambdas: rate of (inserting, deleting, substituting) any token at i
        # predict probabilities: probability of (inserting, substituting) at token i any of the tokens in the dictionary
        batch_size, seq_len = zt.shape
        zt_emb = self.token_embedding(zt)

        time_emb = self.time_embedding(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)

        positions = torch.arange(seq_len, device=zt.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)

        x = zt_emb + time_emb + pos_emb
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        x = x.transpose(0, 1)
        x = self.final_layer_norm(x)

        ins_logits = self.prob_ins(x)
        sub_logits = self.prob_sub(x)
        ins_probs = self.softmax(ins_logits)
        sub_probs = self.softmax(sub_logits)
        rates = self.softplus(self.rates_out(x))

        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        rates = rates * mask_expanded
        ins_probs = ins_probs * mask_expanded
        sub_probs = sub_probs * mask_expanded

        return {"lambdas": {
            "substituting": rates[:,:,0],
            "inserting": rates[:,:,1],
            "deleting": rates[:,:,2],
        }, "probabilities":{
            "substituting": sub_probs,
            "inserting": ins_probs,
        }}
        
    def compute_loss(self, zt, t, z1, sub_mask, ins_mask, del_mask, padding_mask):
        out = self(zt, t, padding_mask)
        loss_sub = sub_mask * (self.mse(out["lambdas"]["substituting"], 1) + self.ce(out["probabilities"]["substituting"], z1))
        loss_ins = ins_mask * (self.mse(out["lambdas"]["inserting"], 1) + self.ce(out["probabilities"]["inserting"], z1))
        loss_del = del_mask * self.mse(out["lambdas"]["deleting"], 1)
        loss = loss_sub + loss_ins + loss_del
        return loss

class TinyShakespeareData(Dataset):
    def __init__(self, data_encoded, seq_len):
        self.data_encoded = data_encoded
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data_encoded) - self.seq_len
    def __getitem__(self, idx):
        return self.data_encoded[idx:idx+self.seq_len]

tiny_shakespeare_data = TinyShakespeareData(data_encoded, seq_len)
train_loader = DataLoader(tiny_shakespeare_data, batch_size=batch_size, shuffle=True)
hidden_dim = 128
num_layers = 3
num_heads = 16
max_seq_len = seq_len
bos_token_id = -1
pad_token_id = PAD_TOKEN
num_epochs = 10
model = NNModel(vocab_size, hidden_dim, num_layers, num_heads, max_seq_len, bos_token_id, pad_token_id)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    x0_batch, x1_batch = get_batch(batch_size, seq_len)
    z1_total = []
    zt_total = []
    sub_mask_total = []
    ins_mask_total = []
    del_mask_total = []
    max_aligned_len = 0
    for i in range(batch_size):
        x0 = x0_batch[i].tolist()
        x1 = x1_batch[i].tolist()
        z0, z1 = align_pair(x0, x1)
        zt, edits, sub_mask, ins_mask, del_mask = get_zt(z0, z1)
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
        z1_total[i] = F.pad(z1_total[i], pad_tuple, "constant", PAD_TOKEN)
        zt_total[i] = F.pad(zt_total[i], pad_tuple, "constant", PAD_TOKEN)
        sub_mask_total[i] = F.pad(sub_mask_total[i], pad_tuple, "constant", 0)
        ins_mask_total[i] = F.pad(ins_mask_total[i], pad_tuple, "constant", 0)
        del_mask_total[i] = F.pad(del_mask_total[i], pad_tuple, "constant", 0)
    z1_total = torch.stack(z1_total)
    zt_total = torch.stack(zt_total)
    sub_mask_total = torch.stack(sub_mask_total)
    ins_mask_total = torch.stack(ins_mask_total)
    del_mask_total = torch.stack(del_mask_total)
    t_batch = torch.rand(batch_size)
    padding_mask = (zt_total == pad_token_id)
    loss = model.compute_loss(zt_total, t_batch, z1_total, sub_mask_total, ins_mask_total, del_mask_total, padding_mask)
    loss = torch.sum(loss) / batch_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

