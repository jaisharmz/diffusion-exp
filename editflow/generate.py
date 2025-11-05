import torch
import torch.nn as nn
from model import NNModel

class Tokenizer:
    def __init__(self, unique_chars):
        unique_chars = sorted(list(unique_chars))
        self.id_to_char = {i : char for i, char in enumerate(unique_chars)}
        self.char_to_id = {char : i for i, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        self.MASK_TOKEN = self.vocab_size
        self.PAD_TOKEN = self.vocab_size + 1
        self.GAP_TOKEN = self.vocab_size + 2
        self.BOS_TOKEN = self.vocab_size + 3
        self.id_to_char[self.MASK_TOKEN] = "<MASK>"
        self.id_to_char[self.PAD_TOKEN] = "<PAD>"
        self.id_to_char[self.GAP_TOKEN] = "<GAP>"
        self.id_to_char[self.BOS_TOKEN] = "<BOS>"
        self.vocab_size += 4
    def __call__(self, string):
        if isinstance(string, list):
            return [[self.BOS_TOKEN] + [self.char_to_id[char] for char in s] for s in string]
        return [self.BOS_TOKEN] + [self.char_to_id[char] for char in string]
    def decode(self, ids):
        if isinstance(ids, list):
            return ["".join(self.id_to_char[id] for id in i) for i in ids]
        return "".join(self.id_to_char[id] for id in ids)
    def to_tensor(self, l):
        # doesn't handle special tokens (e.g. "<PAD>")
        is_string = isinstance(l[0][0], str)
        max_length = max(len(i) for i in l)
        if is_string:
            result = [[i[j] if j < len(i) else self.id_to_char[self.PAD_TOKEN] for j in range(max_length)] for i in l]
        else:
            result = [[i[j] if j < len(i) else self.PAD_TOKEN for j in range(max_length)] for i in l]
        return torch.tensor(result)

def bernoulli_from_rate(rate, tau):
    p = (rate.float() * float(tau)).clamp_(0.0, 1.0 - 1e-6)
    return torch.bernoulli(p)

def sample_from_logits(logits):
    return int(torch.distributions.Categorical(logits=logits).sample().item())

def generate(model, zt, t, padding_mask, tau=0.02):
    rates, sub_logits, ins_logits = model(zt, t, padding_mask)
    sub_rate = rates[:,:,0]
    ins_rate = rates[:,:,1]
    del_rate = rates[:,:,2]
    comb_rate = sub_rate + del_rate
    comb_fire = bernoulli_from_rate(comb_rate, tau).bool()
    p_del = del_rate / comb_rate
    choose_del = (torch.rand_like(p_del) < p_del) & comb_fire
    choose_sub = comb_fire & (~choose_del)
    ins_fire = bernoulli_from_rate(ins_rate, tau).bool()
    result = []
    for i in range(zt.shape[0]):
        # for each sentence
        result_temp = []
        for j in range(zt.shape[1]):
            # for each seq idx
            if choose_sub[i,j]:
                new_idx = sample_from_logits(sub_logits[i,j])
                result_temp.append(new_idx)
            elif not(choose_del[i,j]):
                result_temp.append(int(zt[i,j].item()))
            if ins_fire[i,j]:
                new_idx = sample_from_logits(ins_logits[i,j])
                result_temp.append(new_idx)
        result.append(result_temp)
    return result

if __name__ == "__main__":
    filename = "tinyshakespeare.txt"
    df = open(filename, "r").read()[:1000]
    unique_chars = set(df)
    tokenizer = Tokenizer(unique_chars)
    data_encoded = tokenizer(df)
    print(data_encoded)

    model = NNModel(vocab_size=tokenizer.vocab_size, hidden_dim=64, num_layers=3, num_heads=16, 
                    max_seq_len=256, bos_token_id=tokenizer.BOS_TOKEN, pad_token_id=tokenizer.PAD_TOKEN)
    print(model)
    strings = ["hello world!", "wow this is cool"]
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
