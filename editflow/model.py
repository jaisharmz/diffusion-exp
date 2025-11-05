# === model.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F


class NNModel(nn.Module):
    def __init__(self, vocab_size, dim=256, n_heads=4, n_layers=4, pad_token_id=0):
        super().__init__()
        self.dim = dim
        self.pad_token_id = pad_token_id

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.time_embedding = nn.Linear(1, dim)
        self.position_embedding = nn.Embedding(512, dim)

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads for edit rates
        self.substitution_head = nn.Linear(dim, vocab_size)
        self.insertion_head = nn.Linear(dim, vocab_size)
        self.deletion_head = nn.Linear(dim, 1)

        # FIX: ignore pad tokens in CE loss to prevent gradient noise
        self.ce = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)

    def forward(self, zt, t, padding_mask):
        """
        Forward pass computing substitution/insertion/deletion logits.
        zt: [B, L] token ids at intermediate time t
        t: [B] diffusion timestep (continuous)
        padding_mask: [B, L] boolean mask (True for padding)
        """
        batch, seq_len = zt.shape
        positions = torch.arange(seq_len, device=zt.device).unsqueeze(0)

        # Embeddings
        zt_emb = self.token_embedding(zt)
        time_emb = self.time_embedding(t.view(-1, 1)).unsqueeze(1)
        pos_emb = self.position_embedding(positions)

        x = zt_emb + time_emb + pos_emb

        # Transformer expects [L, B, D]
        x = x.transpose(0, 1)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = x.transpose(0, 1)

        # Heads
        sub_logits = self.substitution_head(x)
        ins_logits = self.insertion_head(x)
        del_logits = self.deletion_head(x)

        return sub_logits, ins_logits, del_logits

    def compute_loss(self, zt, z1, padding_mask, t):
        """
        Computes edit flow loss between zt (noised) and z1 (target) conditioned on t.
        """
        # FIX: ensure float mask for arithmetic operations
        mask = (~padding_mask).float()

        sub_logits, ins_logits, del_logits = self.forward(zt, t, padding_mask)

        # Edit probabilities
        sub_probs = F.softmax(sub_logits, dim=-1)
        ins_probs = F.softmax(ins_logits, dim=-1)
        del_rate = torch.sigmoid(del_logits)

        # Cross-entropy edit losses
        loss_rate_sub = self.ce(sub_logits.transpose(1, 2), z1) * mask
        loss_rate_ins = self.ce(ins_logits.transpose(1, 2), z1) * mask

        # Binary deletion rate loss
        loss_rate_del = F.binary_cross_entropy(
            del_rate.squeeze(-1),
            torch.zeros_like(del_rate.squeeze(-1)),
            reduction="none",
        ) * mask

        # Combine and normalize
        total_rate = loss_rate_sub + loss_rate_ins + loss_rate_del
        # FIX: normalize by number of non-padding tokens instead of batch size
        loss = total_rate.sum() / mask.sum()

        return loss
