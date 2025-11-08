import torch
import torch.nn as nn
import random
from tokenizer import Tokenizer

class Edit:
    def __init__(self, kind, pos, token):
        self.kind = kind
        self.pos = pos
        self.token = token

class CubicKappaScheduler:
    def __init__(self):
        self.a = 1.0
        self.b = 1.0
    def kappa(self, t):
        return (self.a + 1) * (t ** 3) - (self.a + self.b + 1) * (t ** 2) + (self.b + 1) * t
    def kappa_derivative(self, t):
        return 3 * (self.a + 1) * (t ** 2) - 2 * (self.a + self.b + 1) * t + (self.b + 1)
    def weight(self, t):
        return self.kappa_derivative(t) / (1 - self.kappa(t) + 1e-6)

def get_batch(batch_size, seq_len, data_encoded, tokenizer):
    x0 = []
    x1 = []
    for j in range(batch_size):
        i = random.randint(0, len(data_encoded) - seq_len)
        x0_chunk = [tokenizer.MASK_TOKEN for k in range(seq_len)]
        x1_chunk = data_encoded[i:i+seq_len]
        x0.append(x0_chunk)
        x1.append(x1_chunk)
    x0 = tokenizer.to_tensor(x0)
    x1 = tokenizer.to_tensor(x1)
    return x0, x1

def align_pair(x0, x1, tokenizer: Tokenizer):
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
            z0.append(tokenizer.GAP_TOKEN)
            z1.append(x1[j-1])
            j -= 1
        else:
            z0.append(x0[i-1])
            z1.append(tokenizer.GAP_TOKEN)
            i -= 1
    z0 = z0 + [tokenizer.GAP_TOKEN] * i
    z1 = z1 + [tokenizer.GAP_TOKEN] * j
    z0.reverse()
    z1.reverse()
    if z0[0] != tokenizer.BOS_TOKEN:
        z0 = [tokenizer.BOS_TOKEN] + z0
    if z1[0] != tokenizer.BOS_TOKEN:
        z1 = [tokenizer.BOS_TOKEN] + z1

    return {"z0": z0, "z1": z1}

def get_zt(z0, z1, t, tokenizer:Tokenizer):
    # 0 = substitute, 1 = add, 2 = delete
    edit_indices = [i for i in range(len(z0)) if z0[i] != z1[i]]
    # k = random.randint(0, len(edit_indices))
    # indices_to_edit = random.sample(edit_indices, k)
    indices_to_edit = []
    k_t = t ** 3
    for i in edit_indices:
        if random.random() < k_t:
            indices_to_edit.append(i)
    zt = z0[:]
    edits = []
    sub_mask = [0 for i in range(len(z0))]
    ins_mask = [0 for i in range(len(z0))]
    del_mask = [0 for i in range(len(z0))]
    for i in indices_to_edit:
        zt[i] = z1[i]
    for i in set(edit_indices) - set(indices_to_edit):
        edit = 0
        if z1[i] == tokenizer.GAP_TOKEN:
            edit = 2
            del_mask[i] = 1
        elif z0[i] == tokenizer.GAP_TOKEN:
            edit = 1
            ins_mask[i] = 1
        else:
            sub_mask[i] = 1
        edits.append([i, edit])

    sub_mask = torch.tensor(sub_mask)
    ins_mask = torch.tensor(ins_mask)
    del_mask = torch.tensor(del_mask)
    return zt, edits, sub_mask, ins_mask, del_mask

def build_remaining_edits(zt, z1, tokenizer):
    edits = []
    BLANK = tokenizer.BLANK
    def count_nonblank_prefix(z, j):
        c = 0
        for k in range(j):
            if z[k] != BLANK:
                c += 1
        return c
    
    for j, (a, b) in enumerate(zip(zt, z1)):
        if a == b:
            continue
        nb = count_nonblank_prefix(zt, j)
        if a == BLANK and b != BLANK: # insertion
            edits.append(Edit("INS", max(nb - 1, 0), b))
        elif a != BLANK and b == BLANK: # deletion
            edits.append(Edit("DEL", nb, b))
        else: # substitution
            edits.append(Edit("SUB", nb, b))
    return edits

def pad_1d(batch_lists, pad_val):
    B = len(batch_lists)
    Lmax = max((len(x) for x in batch_lists), default=0)
    out = torch.full((B, Lmax), pad_val, dtype=torch.long)
    mask = torch.zeros((B, Lmax), dtype=torch.long)
    for b, x in enumerate(batch_lists):
        if not x:
            continue
        L = len(x)
        out[b, :L] = torch.tensor(x, dtype=torch.long)
        mask[b, :L] = 1
    return out, mask

def perturb_string(s, tokenizer, p=0.3):
    """Randomly substitutes characters in a string with other vocab characters."""
    # Get all valid characters from the tokenizer's vocab, excluding <MASK>
    vocab_chars = [v for k,v in tokenizer.inv_vocab.items() if k != tokenizer.MASK_TOKEN]
    
    chars = list(s)
    for i in range(len(chars)):
        if random.random() < p:
            # Pick a random new character
            new_char = random.choice(vocab_chars)
            chars[i] = new_char
    return "".join(chars)