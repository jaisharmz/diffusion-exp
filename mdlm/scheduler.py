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