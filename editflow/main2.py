import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

# --- 1. Setup (Adapted from your code) ---

filename = "tinyshakespeare.txt"
# Use a bit more data for stable loss
df = open(filename, "r").read()[:10000] 
unique_chars = sorted(list(set(df)))
id_to_char = {i : char for i, char in enumerate(unique_chars)}
char_to_id = {char : i for i, char in enumerate(unique_chars)}
data_encoded = [char_to_id[char] for char in df]
vocab_size = len(unique_chars)

# Add special tokens
PAD_TOKEN = vocab_size
GAP_TOKEN = vocab_size + 1
BOS_TOKEN = vocab_size + 2 # Beginning-of-Sequence, CRITICAL for generation
vocab_size = vocab_size + 3

id_to_char[PAD_TOKEN] = "<PAD>"
id_to_char[GAP_TOKEN] = "<GAP>"
id_to_char[BOS_TOKEN] = "<BOS>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32 # Smaller batch size for faster steps
seq_len = 128
hidden_dim = 128
num_layers = 3
num_heads = 4 # Smaller head count
max_seq_len = seq_len * 2 # Max size for *aligned* sequences
num_steps = 100 # Train for more steps

def get_batch(batch_size, seq_len):
    # Returns x1 (target data) only
    x1 = []
    for batch in range(batch_size):
        # Target with BOS
        start_x1 = [BOS_TOKEN]
        idx = random.randint(0,len(data_encoded) - seq_len)
        # -1 to account for BOS token
        start_x1.extend(data_encoded[idx:idx + seq_len - 1]) 
        x1.append(start_x1)
        
    x1 = torch.tensor(x1, dtype=torch.long)
    return x1

def align_pair(x0, x1):
    # (Your align_pair function is correct, no changes needed)
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
            z1.append(x1[j-1]) # Correction: Add token from x1
            j -= 1
        else: # i > 0 and dp[i][j] == 1 + dp[i-1][j]
            z0.append(x0[i-1]) # Correction: Add token from x0
            z1.append(GAP_TOKEN)
            i -= 1
    z0.reverse()
    z1.reverse()
    return z0, z1

def get_zt(z0, z1, t):
    # (Your get_zt function is correct, no changes needed)
    edit_indices = [i for i in range(len(z0)) if z0[i] != z1[i]]
    indices_to_edit = []
    k_t = t ** 3
    for i in edit_indices:
        if random.random() < k_t:
            indices_to_edit.append(i)
    zt = z0[:]
    sub_mask = [0 for i in range(len(z0))]
    ins_mask = [0 for i in range(len(z0))]
    del_mask = [0 for i in range(len(z0))]
    for i in indices_to_edit:
        zt[i] = z1[i]
    for i in set(edit_indices) - set(indices_to_edit):
        if z1[i] == GAP_TOKEN:
            del_mask[i] = 1
        elif z0[i] == GAP_TOKEN:
            ins_mask[i] = 1
        else:
            sub_mask[i] = 1
    sub_mask = torch.tensor(sub_mask)
    ins_mask = torch.tensor(ins_mask)
    del_mask = torch.tensor(del_mask)
    return zt, sub_mask, ins_mask, del_mask

def safe_log(x):
    return torch.log(x.clamp(min=1e-10))

# --- 2. NEW Helper Functions (The "Stripping" Logic) ---

def rm_gap_tokens(
    z: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Removes gap tokens from a batch of aligned sequences.
    z: (B, L_aligned) - Gappy tensor
    Returns:
    x: (B, L_stripped) - Stripped tensor
    x_pad_mask: (B, L_stripped) - Padding mask for x
    z_gap_mask: (B, L_aligned) - Mask of where gaps were in z
    z_pad_mask: (B, L_aligned) - Mask of where pads were in z
    """
    B, L_aligned = z.shape
    z_pad_mask = (z == PAD_TOKEN)
    z_gap_mask = (z == GAP_TOKEN)
    z_tok_mask = ~z_gap_mask & ~z_pad_mask

    # 1. Get the 1D tensor of all valid tokens, concatenated
    valid_tokens = z[z_tok_mask] # This is the tensor of size 4092

    # 2. Calculate the new, stripped lengths for each sequence
    x_lens = z_tok_mask.sum(dim=1) # Shape: [B]
    
    # 3. Find the max stripped length
    max_x_len = x_lens.max() # This will be <= L_aligned
    
    # 4. Create the new (empty) x tensor, pre-filled with PAD_TOKENs
    x = torch.full((B, max_x_len), PAD_TOKEN, device=z.device, dtype=torch.long)
    
    # 5. Create the padding mask for x
    # x_pad_mask[b, j] is True if j >= x_lens[b]
    x_pad_mask = torch.arange(max_x_len, device=z.device)[None, :] >= x_lens[:, None]
    
    # 6. Fill the x tensor
    # ~x_pad_mask is a 2D mask of shape [B, max_x_len]
    # The number of True elements in it is x_lens.sum(), which is 4092
    # We can directly assign the 1D valid_tokens to the True positions
    x[~x_pad_mask] = valid_tokens
    
    return x, x_pad_mask, z_gap_mask, z_pad_mask


def build_remaining_edits(
    zt: torch.Tensor, 
    z1: torch.Tensor, 
    z_gap_mask: torch.Tensor, 
    z_pad_mask: torch.Tensor,
    sub_mask_gappy: torch.Tensor, 
    ins_mask_gappy: torch.Tensor, 
    del_mask_gappy: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Maps the gappy edit masks to the coordinates of the stripped sequence xt.
    """
    B, L_aligned = zt.shape
    
    # Get non-gap, non-pad token positions
    z_tok_mask = ~z_gap_mask & ~z_pad_mask
    
    # cumsum gives us the mapping from aligned (gappy) space to stripped space
    # z_tok_idx[b, i] = k means zt[b, i] corresponds to xt[b, k]
    # We subtract 1 because cumsum is 1-indexed
    z_tok_idx = (z_tok_mask.cumsum(dim=1) - 1).long()

    # --- Map Deletions and Substitutions ---
    # These happen at tokens that *exist* in xt
    
    # 1. Deletions
    # Find where gappy del_mask is 1 AND the token exists in xt
    del_mask_stripped_op = (del_mask_gappy == 1) & z_tok_mask
    # Get the batch indices (rows) and aligned indices (cols)
    b_del, z_idx_del = torch.where(del_mask_stripped_op)
    # Map gappy z_idx to stripped x_idx
    x_idx_del = z_tok_idx[b_del, z_idx_del]

    # 2. Substitutions
    sub_mask_stripped_op = (sub_mask_gappy == 1) & z_tok_mask
    b_sub, z_idx_sub = torch.where(sub_mask_stripped_op)
    x_idx_sub = z_tok_idx[b_sub, z_idx_sub]
    # Get the target token from z1
    tok_sub = z1[b_sub, z_idx_sub]

    # --- Map Insertions ---
    # These happen *before* tokens that exist in xt.
    # We map an insertion at zt[i] to the position *before* xt[k]
    
    ins_mask_stripped_op = (ins_mask_gappy == 1)
    b_ins, z_idx_ins = torch.where(ins_mask_stripped_op)
    
    # The insertion position in xt is the index of the *next* token in zt
    # We find the next token by searching forward from z_idx_ins
    
    x_idx_ins = torch.zeros_like(b_ins)
    tok_ins = z1[b_ins, z_idx_ins] # Get the token to be inserted

    for i in range(len(b_ins)):
        b = b_ins[i]
        z_idx = z_idx_ins[i]
        
        # Find the index of the *next* non-gap token in this row
        next_tok_mask = z_tok_mask[b, z_idx+1:]
        
        if next_tok_mask.any():
            # If a token exists after, insertion happens before it
            z_idx_next_tok = z_idx + 1 + torch.where(next_tok_mask)[0][0]
            x_idx_ins[i] = z_tok_idx[b, z_idx_next_tok]
        else:
            # No token exists after. Insert at the end.
            # This is the length of the stripped sequence
            x_idx_ins[i] = max(0, z_tok_mask[b].sum() - 1)

    return {
        "b_sub": b_sub, "x_idx_sub": x_idx_sub, "tok_sub": tok_sub,
        "b_ins": b_ins, "x_idx_ins": x_idx_ins, "tok_ins": tok_ins,
        "b_del": b_del, "x_idx_del": x_idx_del,
    }

# --- 3. Refactored NNModel ---

class SinusoidalTimeEmbedding(nn.Module):
    # (Your SinusoidalTimeEmbedding is correct, no changes)
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
        self.pad_token_id = pad_token_id
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim))
        
        # NOTE: max_seq_len here is for the *stripped* sequence,
        # but we use a larger value just in case.
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim) 
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
                                       dropout=0.1, activation="gelu", batch_first=True) # Use batch_first=True
            for i in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        
        # Rates for Sub/Del (at a token)
        self.rates_sub_del = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2) # 0=sub, 1=del
        )
        # Rates for Ins (between tokens)
        self.rates_ins = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # 0=ins
        )
        
        # Probs for Sub/Ins
        self.prob_sub = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.prob_ins = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.softplus = nn.Softplus()

    def forward(self, xt, t, xt_pad_mask):
        # xt is (B, L_stripped)
        # xt_pad_mask is (B, L_stripped)
        
        batch_size, seq_len = xt.shape
        
        # Get embeddings
        xt_emb = self.token_embedding(xt)
        time_emb = self.time_embedding(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        positions = torch.arange(seq_len, device=xt.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)

        x = xt_emb + time_emb + pos_emb
        
        # Transformer
        # We use xt_pad_mask as the key_padding_mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=xt_pad_mask)
        x = self.final_layer_norm(x) # (B, L, D)

        # --- Predict Rates ---
        # Rates for Sub/Del are predicted *at* each token
        rates_sub_del = self.softplus(self.rates_sub_del(x)) # (B, L, 2)
        sub_rate = rates_sub_del[..., 0]
        del_rate = rates_sub_del[..., 1]
        
        # Rates for Ins are predicted *at* each token (representing "insert before")
        ins_rate = self.softplus(self.rates_ins(x)).squeeze(-1) # (B, L)
        
        # --- Predict Probabilities ---
        sub_logits = self.prob_sub(x) # (B, L, C)
        ins_logits = self.prob_ins(x) # (B, L, C)

        # Apply padding mask to all outputs
        mask_expanded = (~xt_pad_mask).unsqueeze(-1).float()
        
        return {"lambdas": {
            "substituting": sub_rate * ~xt_pad_mask,
            "inserting": ins_rate * ~xt_pad_mask, # ins_rate[i] = insert *before* xt[i]
            "deleting": del_rate * ~xt_pad_mask,
        }, "probabilities":{
            "substituting": sub_logits * mask_expanded,
            "inserting": ins_logits * mask_expanded,
        }}
        
    def compute_loss(self, xt, t, xt_pad_mask, edits: Dict[str, torch.Tensor]):
        """
        Computes the NLL loss in the STRIPPED space.
        """
        out = self(xt, t, xt_pad_mask)
        
        B, L, C = out["probabilities"]["substituting"].shape
        
        sub_rate = out["lambdas"]["substituting"]
        ins_rate = out["lambdas"]["inserting"]
        del_rate = out["lambdas"]["deleting"]
        
        sub_logits = out["probabilities"]["substituting"]
        ins_logits = out["probabilities"]["inserting"]
        
        # --- 1. Survival Loss (The "Push") ---
        # Sum of all rates at all non-padded positions
        total_rate = (sub_rate + ins_rate + del_rate) * (~xt_pad_mask)
        loss_surv = total_rate.sum(dim=1) # (B,)

        # --- 2. Positive Loss (The "Pull") ---
        # We need to calculate the total positive NLL *per batch item*
        # This tensor will store the positive loss for each sample.
        loss_pos_per_sample = torch.zeros(B, device=xt.device, dtype=torch.float32)

        # A. Substitution Loss
        b_sub, x_idx_sub, tok_sub = edits["b_sub"], edits["x_idx_sub"], edits["tok_sub"]
        if b_sub.numel() > 0:
            # Get the rate at the correct (batch, seq_pos)
            rate_sub_pos = sub_rate[b_sub, x_idx_sub]
            # Get the logits for the correct (batch, seq_pos)
            logits_sub_pos = sub_logits[b_sub, x_idx_sub] # (N_sub, C)
            # Calculate NLL for the target token
            loss_ce_sub = self.ce(logits_sub_pos, tok_sub)
            
            # Total NLL for each substitution edit
            nll_sub = -safe_log(rate_sub_pos) + loss_ce_sub # Shape: (N_sub,)
            
            # Add these losses to the correct batch items
            # This is the critical step:
            loss_pos_per_sample.scatter_add_(0, b_sub, nll_sub)
        
        # B. Insertion Loss
        b_ins, x_idx_ins, tok_ins = edits["b_ins"], edits["x_idx_ins"], edits["tok_ins"]
        if b_ins.numel() > 0:
            # Get rate at (batch, seq_pos_to_insert_before)
            rate_ins_pos = ins_rate[b_ins, x_idx_ins]
            # Get logits at (batch, seq_pos_to_insert_before)
            logits_ins_pos = ins_logits[b_ins, x_idx_ins] # (N_ins, C)
            # Calculate NLL for the target token
            loss_ce_ins = self.ce(logits_ins_pos, tok_ins)
            
            # Total NLL for each insertion edit
            nll_ins = -safe_log(rate_ins_pos) + loss_ce_ins # Shape: (N_ins,)
            
            # Add these losses to the correct batch items
            loss_pos_per_sample.scatter_add_(0, b_ins, nll_ins)
            
        # C. Deletion Loss
        b_del, x_idx_del = edits["b_del"], edits["x_idx_del"]
        if b_del.numel() > 0:
            rate_del_pos = del_rate[b_del, x_idx_del]
            
            # Total NLL for each deletion edit
            nll_del = -safe_log(rate_del_pos) # Shape: (N_del,)
            
            # Add these losses to the correct batch items
            loss_pos_per_sample.scatter_add_(0, b_del, nll_del)
            
        # loss_pos is now correctly shaped (B,)
        loss_pos = loss_pos_per_sample 

        # --- 3. Time-Weighted Final Loss ---
        kappa_t = 3 * t**2    # (B,)
        one_minus_t = 1.0 - t # (B,)
        
        # This is now a correct element-wise operation (B,) * (B,)
        final_loss_per_sample = (one_minus_t * loss_surv) + (kappa_t * loss_pos) # (B,)
        
        # Return the mean loss over the batch
        return final_loss_per_sample.mean()

    @torch.no_grad()
    def generate(self, n_samples, n_steps, device, bos_token_id, pad_token_id):
        """
        Generation in the "stripped" space.
        Starts with <BOS> and inserts tokens.
        """
        self.eval()
        
        # Start with just the BOS token
        xt = torch.full((n_samples, 1), bos_token_id, device=device, dtype=torch.long)
        
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = i * dt
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.float32)
            
            xt_pad_mask = (xt == pad_token_id)
            
            out = self(xt, t_batch, xt_pad_mask)
            
            # Get rates and logits
            lambda_sub = out["lambdas"]["substituting"]
            lambda_ins = out["lambdas"]["inserting"]
            lambda_del = out["lambdas"]["deleting"]
            sub_logits = out["probabilities"]["substituting"]
            ins_logits = out["probabilities"]["inserting"]

            # --- Sample Edits ---
            # P(event) = 1 - exp(-rate * dt)
            p_sub = 1.0 - torch.exp(-lambda_sub * dt)
            p_ins = 1.0 - torch.exp(-lambda_ins * dt)
            p_del = 1.0 - torch.exp(-lambda_del * dt)
            
            # Sample event masks
            sub_event_mask = torch.rand_like(p_sub) < p_sub
            ins_event_mask = torch.rand_like(p_ins) < p_ins
            del_event_mask = torch.rand_like(p_del) < p_del
            
            # Apply padding mask (don't edit padding)
            sub_event_mask &= ~xt_pad_mask
            ins_event_mask &= ~xt_pad_mask
            del_event_mask &= ~xt_pad_mask
            
            # Don't delete BOS token
            del_event_mask[:, 0] = 0 
            
            # --- Apply Edits ---
            # This is complex because inserts/deletes change sequence length.
            # We process one sample at a time.
            
            new_xt_list = []
            for b in range(n_samples):
                new_seq = []
                old_seq = xt[b, ~xt_pad_mask[b]] # Get non-pad tokens
                
                # Iterate through old_seq, deciding what to do at each token xt[i]
                for i in range(len(old_seq)):
                    # 1. Insertion *before* xt[i]
                    if ins_event_mask[b, i]:
                        ins_tok = torch.distributions.Categorical(logits=ins_logits[b, i]).sample()
                        new_seq.append(ins_tok.item())
                    
                    # 2. Deletion or Substitution *at* xt[i]
                    # (Prioritize deletion over substitution)
                    if del_event_mask[b, i]:
                        pass # Don't append old_seq[i]
                    elif sub_event_mask[b, i]:
                        sub_tok = torch.distributions.Categorical(logits=sub_logits[b, i]).sample()
                        new_seq.append(sub_tok.item())
                    else:
                        new_seq.append(old_seq[i].item()) # Keep token

                new_xt_list.append(torch.tensor(new_seq, device=device, dtype=torch.long))

            # Pad all sequences in the batch to the new max_len
            xt = torch.nn.utils.rnn.pad_sequence(new_xt_list, batch_first=True, padding_value=pad_token_id)

            # Safety break if sequences get too long
            if xt.shape[1] > max_seq_len:
                break
                
        self.train()
        
        # --- Clean up final sequences ---
        final_sequences = []
        for i in range(n_samples):
            seq_tensor = xt[i]
            # Filter out pad tokens and BOS token
            final_seq = seq_tensor[
                (seq_tensor != pad_token_id) & 
                (seq_tensor != bos_token_id)
            ]
            final_sequences.append(final_seq)
            
        return final_sequences


# --- 4. Refactored Training Loop ---

model = NNModel(
    vocab_size, hidden_dim, num_layers, num_heads, 
    max_seq_len, BOS_TOKEN, PAD_TOKEN
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = []
print("Starting training...")

for step in range(num_steps):
    # 1. Get Gappy Data (z0, z1, zt)
    
    # Get x1 (target data) from the new get_batch
    x1_batch = get_batch(batch_size, seq_len) # (B, L)
    
    # Manually create x0 (base distribution) as a list of [BOS_TOKEN]
    x0_list = [[BOS_TOKEN] for _ in range(batch_size)]

    z1_gappy = []
    zt_gappy = []
    sub_mask_gappy = []
    ins_mask_gappy = []
    del_mask_gappy = []
    max_aligned_len = 0
    
    t_batch = torch.rand(batch_size).to(device)

    for i in range(batch_size):
        x0 = x0_list[i] # x0 is now just [BOS_TOKEN]
        x1 = x1_batch[i].tolist() # x1 is [BOS, 'F', 'i', 'r', ...]
        
        # This will align [BOS_TOKEN] with a full sentence
        # z0 becomes [BOS, <GAP>, <GAP>, ...]
        # z1 becomes [BOS, 'F', 'i', 'r', ...]
        z0, z1 = align_pair(x0, x1)
        
        t_sample = t_batch[i].item()
        zt, sub_mask, ins_mask, del_mask = get_zt(z0, z1, t_sample)
        
        z1_gappy.append(torch.tensor(z1, dtype=torch.long))
        zt_gappy.append(torch.tensor(zt, dtype=torch.long))
        sub_mask_gappy.append(sub_mask)
        ins_mask_gappy.append(ins_mask)
        del_mask_gappy.append(del_mask)

        if len(zt) > max_aligned_len:
            max_aligned_len = len(zt)

    # Pad gappy tensors
    for i in range(batch_size):
        pad_len = max_aligned_len - zt_gappy[i].shape[0]
        pad_tuple = (0, pad_len)
        zt_gappy[i] = F.pad(zt_gappy[i], pad_tuple, "constant", PAD_TOKEN)
        z1_gappy[i] = F.pad(z1_gappy[i], pad_tuple, "constant", PAD_TOKEN)
        sub_mask_gappy[i] = F.pad(sub_mask_gappy[i], pad_tuple, "constant", 0)
        ins_mask_gappy[i] = F.pad(ins_mask_gappy[i], pad_tuple, "constant", 0)
        del_mask_gappy[i] = F.pad(del_mask_gappy[i], pad_tuple, "constant", 0)

    zt_gappy = torch.stack(zt_gappy).to(device)
    z1_gappy = torch.stack(z1_gappy).to(device)
    sub_mask_gappy = torch.stack(sub_mask_gappy).to(device)
    ins_mask_gappy = torch.stack(ins_mask_gappy).to(device)
    del_mask_gappy = torch.stack(del_mask_gappy).to(device)
    
    # --- 2. Convert to Stripped Data ---
    
    # A. Strip zt (model input)
    xt_total, xt_pad_mask, zt_gap_mask, zt_pad_mask = rm_gap_tokens(zt_gappy)
    
    # B. Map the edit masks
    edits = build_remaining_edits(
        zt_gappy, z1_gappy, zt_gap_mask, zt_pad_mask,
        sub_mask_gappy, ins_mask_gappy, del_mask_gappy
    )
    
    # --- 3. Compute Loss with Stripped Data ---
    
    # Pass xt_total (stripped zt) to the model
    loss = model.compute_loss(xt_total, t_batch, xt_pad_mask, edits)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Grad clipping
    optimizer.step()
    
    losses.append(loss.item())
    
    if step % 1 == 0 or step == num_steps - 1:
        print(f"Step {step}: Loss {loss.item():.4f}")

print("Training finished.")

# --- 5. Plot Loss ---

plt.plot(torch.arange(len(losses)), losses)
plt.title("Training Loss (Stripped NLL Loss)")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()

# --- 6. Generation ---

print("--- Generating Samples ---")
generated_sequences = model.generate(
    n_samples=4,
    n_steps=100,
    device=device,
    bos_token_id=BOS_TOKEN,
    pad_token_id=PAD_TOKEN
)

for i, seq_tensor in enumerate(generated_sequences):
    # Decode to text
    text = "".join([id_to_char[token_id.item()] for token_id in seq_tensor])
    print(f"Sample {i}: {text}\n")