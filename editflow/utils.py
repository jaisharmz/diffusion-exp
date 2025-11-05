import torch
import torch.nn as nn
import random
from generate import Tokenizer

def get_batch(batch_size, seq_len, data_encoded, tokenizer):
    x0 = []
    x1 = []
    for j in range(batch_size):
        i = random.randint(0, len(data_encoded) - seq_len - 5)
        shift = random.randint(1, 4)
        x0_chunk = data_encoded[i : i + seq_len]
        x1_chunk = data_encoded[i + shift : i + seq_len + shift]
        x0.append([tokenizer.BOS_TOKEN] + x0_chunk)
        x1.append([tokenizer.BOS_TOKEN] + x1_chunk)
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
    return z0, z1

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