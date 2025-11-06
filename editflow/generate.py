import torch
import torch.nn as nn
from model import NNModel
from tokenizer import Tokenizer

def bernoulli_from_rate(rate, tau):
    p = (rate.float() * float(tau)).clamp_(0.0, 1.0 - 1e-6)
    return torch.bernoulli(p)

def sample_from_logits(logits):
    return int(torch.distributions.Categorical(logits=logits).sample().item())

def generate(model, zt, attention_mask, t, tau=0.02):
    model.eval()
    with torch.no_grad():
        out = model(input_ids=zt, attention_mask=attention_mask, t=t)
        sub_rate = out["sub_rate"]
        del_rate = out["del_rate"] 
        ins_rate = out["ins_rate"]
        sub_logits = out["sub_logits"]
        ins_logits = out["ins_logits"]
        
        comb_rate = sub_rate + del_rate
        comb_fire = bernoulli_from_rate(comb_rate, tau).bool()
        p_del = del_rate / (comb_rate + 1e-9)
        choose_del = (torch.rand_like(p_del) < p_del) & comb_fire
        choose_sub = comb_fire & (~choose_del)
        ins_fire = bernoulli_from_rate(ins_rate, tau).bool()

        result = []
        for i in range(zt.shape[0]):  # For each sequence in the batch
            result_temp = []
            for j in range(zt.shape[1]): # For each token position
                if choose_sub[i, j]:
                    new_idx = sample_from_logits(sub_logits[i, j])
                    result_temp.append(new_idx)
                elif not choose_del[i, j]:
                    if attention_mask[i, j]:
                        result_temp.append(int(zt[i, j].item()))
                if ins_fire[i, j]:
                    new_idx = sample_from_logits(ins_logits[i, j])
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
        attention_mask = zt != tokenizer.PAD_TOKEN
        zt = generate(model, zt, attention_mask, t)
        strings = tokenizer.decode(zt)
        print(strings)
        print(zt)
