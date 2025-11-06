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