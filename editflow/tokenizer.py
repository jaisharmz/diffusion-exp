import torch
import torch.nn as nn

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
        self.BLANK = -1
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

# class Tokenizer:
#     def __init__(self):
#         self.vocab = {"<MASK>": 0}
#         self.inv_vocab = {0: "<MASK>"}
#         self.MASK_TOKEN = 0
#         self.vocab_size = len(self.vocab.keys())

#     def __call__(self, text):
#         for ch in text:
#             if ch not in self.vocab:
#                 idx = len(self.vocab)
#                 self.vocab[ch] = idx
#                 self.inv_vocab[idx] = ch
#         self.vocab_size = len(self.vocab.keys())
#         return [self.vocab[ch] for ch in text]
    
#     def encode(self, text):
#         return self(text)

#     def decode(self, tokens):
#         return "".join([self.inv_vocab.get(t, "") for t in tokens if t not in (self.MASK_TOKEN)])

#     def to_tensor(self, batch):
#         max_len = max(len(seq) for seq in batch)
#         tensor = torch.full((len(batch), max_len), self.MASK_TOKEN, dtype=torch.long)
#         for i, seq in enumerate(batch):
#             tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
#         return tensor