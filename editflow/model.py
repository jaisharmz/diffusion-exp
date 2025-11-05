import torch
import torch.nn as nn

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
        self.prob_sub = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.mse = nn.MSELoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
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
        # ins_probs = self.softmax(ins_logits)
        # sub_probs = self.softmax(sub_logits)
        rates = self.softplus(self.rates_out(x))

        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        rates = rates * mask_expanded
        # ins_probs = ins_probs * mask_expanded
        # sub_probs = sub_probs * mask_expanded

        return rates, sub_logits, ins_logits
        
    def compute_loss(self, zt, t, z1, sub_mask, ins_mask, del_mask, padding_mask):
        mask = (~padding_mask).float()
        rates, sub_logits, ins_logits = self(zt, t, padding_mask)
        sub_rate = rates[:,:,0]
        ins_rate = rates[:,:,1]
        del_rate = rates[:,:,2]
        sub_probs = F.softmax(sub_logits, dim=-1)
        ins_probs = F.softmax(ins_logits, dim=-1)
        del_rate = torch.sigmoid(del_logits)
        total_rate = (sub_rate + ins_rate + del_rate) * (~padding_mask).float()
        loss_surv = total_rate.sum(dim=1)
        loss_ce_sub = self.ce(sub_logits.transpose(1, 2), z1)
        loss_rate_sub = -torch.log(sub_rate + 1e-10)
        loss_sub = sub_mask * (loss_ce_sub + loss_rate_sub)
        loss_ce_ins = self.ce(ins_logits.transpose(1, 2), z1)
        loss_rate_ins = -torch.log(ins_rate + 1e-10)
        loss_ins = ins_mask * (loss_ce_ins + loss_rate_ins)
        loss_rate_del = -torch.log(del_rate + 1e-10)
        loss_del = del_mask * loss_rate_del
        loss_pos = (loss_sub + loss_ins + loss_del).sum(dim=1)
        kappa_t = 3 * t**2
        one_minus_t = 1.0 - t
        final_loss_per_sample = (one_minus_t * loss_surv) + (kappa_t * loss_pos)
        return final_loss_per_sample