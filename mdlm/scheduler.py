import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAlphaScheduler:
    def __call__(self, i):
        return self.alpha(i)
    def alpha(self, i):
        return 1 - i
    def alpha_derivative(self, i):
        return -torch.ones_like(i)
    def weight(self, i):
        return self.alpha_derivative(i) / (1 - self.alpha(i) + 1e-6)
    def reverse_mask_prob(self, s, t):
        t_t = torch.as_tensor(t, dtype=torch.float32, device=t.device if isinstance(t, torch.Tensor) else None)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=t.device if isinstance(t, torch.Tensor) else None)
        out = (1 - self(s_t)) / (1 - self(t_t))
        return out.item() if isinstance(t, float) and isinstance(s, float) else out