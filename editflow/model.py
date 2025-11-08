import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import align_pair, CubicKappaScheduler, build_remaining_edits, pad_1d

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
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, max_seq_len, bos_token_id, pad_token_id, tokenizer, time_epsilon=1e-3, device="cpu"):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.time_epsilon = time_epsilon
        self.device = device
        self.scheduler = CubicKappaScheduler()
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
        self.prob_sub = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.mse = nn.MSELoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.cache_inputs = None
    # def forward(self, zt, t, padding_mask):
    def forward(self, input_ids, attention_mask, t):
        # predict lambdas: rate of (substituting, deleting, inserting) any token at i
        # predict probabilities: probability of (substituting, inserting) at token i any of the tokens in the dictionary
        batch_size, seq_len = input_ids.shape
        zt_emb = self.token_embedding(input_ids)
        time_emb = self.time_embedding(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)

        x = zt_emb + time_emb + pos_emb

        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=(~attention_mask.bool()))
        x = x.transpose(0, 1)
        x = self.final_layer_norm(x)

        ins_logits = self.prob_ins(x)
        sub_logits = self.prob_sub(x)
        rates = self.softplus(self.rates_out(x))

        mask_expanded = (attention_mask).unsqueeze(-1).float()
        rates = rates * mask_expanded
        out = {"sub_rate": rates[:,:,0], "del_rate": rates[:,:,1], "ins_rate": rates[:,:,2],
               "sub_logits": sub_logits, "ins_logits": ins_logits}
        return out
    def compute_loss(self, inputs):
        # inputs have x0 and x1
        device = self.device
        B = len(inputs["x0_ids"])
        use_cache = inputs == self.cache_inputs
        if use_cache:
            aligns = self.cache_aligns
        else:
            aligns = [align_pair(x0, x1, self.tokenizer) for x0, x1 in zip(inputs["x0_ids"], inputs["x1_ids"])]
            self.cache_inputs = inputs
            self.cache_aligns = aligns

        z0_list = [a["z0"] for a in aligns]
        z1_list = [a["z1"] for a in aligns]
        
        
        if use_cache:
            t = self.cache_t
            k = self.scheduler.kappa(t).to(device)
            w = self.scheduler.weight(t).squeeze(1).to(device)
            zt_list = self.cache_zt_list
        else:
            t = (1 - self.time_epsilon) * torch.rand(B, 1, device=device)
            k = self.scheduler.kappa(t).to(device)
            w = self.scheduler.weight(t).squeeze(1).to(device)
            zt_list = []
            for z0, z1, kb in zip(z0_list, z1_list, k.squeeze(1).tolist()):
                choose_target = torch.rand(len(z0)) < kb
                zt = [b if choose_target[j] else a for j, (a, b) in enumerate(zip(z0, z1))]
                zt_list.append(zt)
            self.cache_zt_list = zt_list
            self.cache_t = t
        
        xt_list = [[c for c in zt if c != self.tokenizer.BLANK] for zt in zt_list]
        edits_list = [build_remaining_edits(zt, z1, self.tokenizer) for zt, z1 in zip(zt_list, z1_list)]

        x_tok, x_mask = pad_1d(xt_list, pad_val=self.tokenizer.PAD_TOKEN)
        x_tok, x_mask = x_tok.to(device), x_mask.to(device)
        
        out = self(input_ids=x_tok, attention_mask=x_mask, t=t)
        sub_rate = out["sub_rate"]
        del_rate = out["del_rate"]
        ins_rate = out["ins_rate"]
        logprob_sub = F.log_softmax(out["sub_logits"], dim=-1)
        logprob_ins = F.log_softmax(out["ins_logits"], dim=-1)

        L1 = torch.tensor([len(x1) for x1 in inputs["x1_ids"]], device=device, dtype=torch.float).clamp_min(1.0)
        total_rate = sub_rate + del_rate + ins_rate
        loss_surv = ((w * (total_rate * x_mask.float()).sum(dim=1)) / L1).mean()

        loss_pos_per = sub_rate.new_zeros(B)
        for b, edits in enumerate(edits_list):
            if not edits:
                continue
            cur_len = int(x_mask[b].sum().item())
            for e in edits:
                pos = e.pos
                tok = e.token
                if e.kind == "SUB":
                    loss_pos_per[b] -= logprob_sub[b, pos, tok] + torch.log(sub_rate[b, pos])
                elif e.kind == "DEL":
                    loss_pos_per[b] -= torch.log(del_rate[b, pos])
                elif e.kind == "INS":
                    loss_pos_per[b] -= logprob_ins[b, pos, tok] + torch.log(ins_rate[b, pos])
        loss_pos = ((w * loss_pos_per) / L1).mean()
        loss = loss_surv + loss_pos
        return loss, out