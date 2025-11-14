import torch
import torch.nn.functional as F
import numpy as np
import math

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps, scheduler, stochastic):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps-1, -1, -1), range(steps)):
            s /= steps
            t /= steps
            reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i,0].to(torch.float64) * reverse_transfer_prob
                num_transfer_tokens[i,j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i,0].to(torch.float64)
                num_transfer_tokens[i,j] = torch.distributions.Binomial(n, reverse_transfer_prob).sample().to(torch.int64)
            # num_transfer_tokens[i,j] = torch.minimum(num_transfer_tokens[i,j], mask_num[i,0])
            mask_num[i,0] -= num_transfer_tokens[i,j]
            if mask_num[i,0].item() == 0:
                break
    rows = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, nonzero.numel())
    padded_rows = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad])
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)

class Generator:
    def __init__(self, model, tokenizer, scheduler):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = scheduler
    @torch.no_grad()
    def generate(self, inputs, config, return_dict_in_generate):
        steps = config.steps
        max_new_tokens = config.max_new_tokens
        max_length = config.max_length
        block_length = config.block_length
        temperature = config.temperature
        cfg_scale = config.cfg_scale
        cfg_keep_tokens = config.cfg_keep_tokens
        remasking = config.remasking
        stochastic_transfer = config.stochastic_transfer

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs]
        prompt_lens = [p.shape[0] for p in inputs]
        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)
        
        B = len(inputs)
        T = max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, :prompt_lens[i]] = p
            x[i, prompt_lens[i]:prompt_lens[i]+max_new_tokens] = mask_id
        attention_mask = (x != eos_id).long() if B > 1 else None
        unmasked_index = (x != mask_id) & (x != eos_id)
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=self.model.device))
            unmasked_index = unmasked_index & ~keep_mask
        num_blocks = math.ceil(max_new_tokens / block_length)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict_in_generate else None

        for b in range(num_blocks):
            block_mask_index = torch.zeros((B, block_length), dtype=torch.bool, device=x.device)
            for j in range(B):
                start = prompt_lens[j] + b * block_length
                end = min(start + block_length, T)
                if start < end:
                    width = end - start
                    block_mask_index[j,:width] = x[j,start:end] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(mask_index=block_mask_index, steps=steps, scheduler=self.scheduler, stochastic=stochastic_transfer)
            effective_steps = num_transfer_tokens.size(1)
            for i in range(effective_steps):
                mask_index = x == mask_id
                if cfg_scale > 0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_, attention_mask=attention_mask).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                
                for j in range(B):
                    x0_p[j,prompt_lens[j] + (b + 1) * block_length:] = -np.inf
                x0 = torch.where(mask_index, x0_p, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j,i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())
        if not return_dict_in_generate:
            return x
        else:
            return {"sequences":x, "histories":histories}
    def infill(self):
        pass