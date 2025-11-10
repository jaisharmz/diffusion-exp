import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_batch, align_pair, get_zt, Edit, perturb_string
from generate import Tokenizer, generate
from model import NNModel

# TODO: 1) fix pad token, 2) safe log, 3) make torch Dataset object and collate function, 4) save model

filename = "tinyshakespeare.txt"
df = open(filename, "r").read()#[:1000]
unique_chars = set(df)
tokenizer = Tokenizer()
data_encoded = tokenizer(df)
device = "cuda"

model = NNModel(vocab_size=tokenizer.vocab_size, hidden_dim=64, num_layers=3, num_heads=16, 
                max_seq_len=256, bos_token_id=tokenizer.BOS_TOKEN, pad_token_id=tokenizer.PAD_TOKEN,
                tokenizer=tokenizer, device=device)
model.to(device)

num_steps = 1000
# batch_size = 2
max_seq_len = 128
num_ministeps = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = []
for step in range(num_steps):
    batch_size = max(int(step / num_steps * 128), 1)
    x0_batch, x1_batch = get_batch(batch_size, max_seq_len, data_encoded, tokenizer)
    inputs = {"x0_ids": x0_batch, "x1_ids": x1_batch}
    for ministep in range(num_ministeps):
        loss, out = model.compute_loss(inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if ministep % 1 == 0 or ministep == num_ministeps - 1:
            print(f"Step {step}, Ministep {ministep}: Loss {loss.item():.4f}")
        # if ministep % 10 == 0 or ministep == num_ministeps - 1:
        #     strings = ["hello world!", "wow this is cool", ""]
        #     zt = tokenizer(strings)
        #     for i in range(10):
        #         zt = tokenizer.to_tensor(zt).to(device)
        #         batch_size, seq_len = zt.shape
        #         t = torch.rand(batch_size).to(device)
        #         attention_mask = zt != tokenizer.PAD_TOKEN
        #         zt = generate(model, zt, attention_mask, t)
        #         strings = tokenizer.decode(zt)
        #         print(strings)
        #         print(zt)
        if ministep % 10 == 0 or ministep == num_ministeps - 1:
            print("\n--- Running 'Fix-It' Evaluation ---")
            
            # --- 1. Validation Set (Fixed Holdout) ---
            val_strings = [
                "hello world!", 
                "the quick brown fox"
            ]
            print(f"VAL | Ground Truth: {val_strings}")
            
            # Perturb them
            perturbed_val = [perturb_string(s, tokenizer, p=0.15) for s in val_strings]
            print(f"VAL | Perturbed:    {perturbed_val}")

            # Run the model
            zt_val = tokenizer(perturbed_val)
            zt_val = tokenizer.to_tensor(zt_val).to(device)
            b_val, s_val = zt_val.shape
            
            # --- THIS IS THE FIX ---
            mask_val = zt_val != tokenizer.MASK_TOKEN 
            # ---------------------
            
            t_val = torch.full((b_val,), 0.05, device=device)
            fixed_zt_val = generate(model, zt_val, mask_val, t_val, mode="sub_only") # Use sub_only here
            fixed_strings_val = tokenizer.decode(fixed_zt_val)
            print(f"VAL | Model's Fix:  {fixed_strings_val}\n")

            train_ids = inputs["x1_ids"][:2] # Using just 2 to keep logs clean
            train_strings = tokenizer.decode(train_ids.tolist())
            
            print(f"TRAIN | Ground Truth: {train_strings}")

            # Perturb them
            perturbed_train = [perturb_string(s, tokenizer, p=0.15) for s in train_strings]
            print(f"TRAIN | Perturbed:    {perturbed_train}")
            
            # --- End of missing code ---

            # Run the model
            zt_train = tokenizer(perturbed_train)
            zt_train = tokenizer.to_tensor(zt_train).to(device)
            b_train, s_train = zt_train.shape

            # --- THIS IS THE FIX ---
            mask_train = zt_train != tokenizer.MASK_TOKEN
            # ---------------------

            t_train = torch.full((b_train,), 0.05, device=device)
            fixed_zt_train = generate(model, zt_train, mask_train, t_train, mode="sub_only") # Use sub_only here
            fixed_strings_train = tokenizer.decode(fixed_zt_train)
            print(f"TRAIN | Model's Fix:  {fixed_strings_train}\n")