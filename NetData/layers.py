import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, num_embed):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(num_embed, num_embed),
            nn.Tanh(),
            nn.Linear(num_embed, 1, bias=False)
        )

    def forward(self, z, mask=None):
        """
        z:    [B, T, D]
        mask: [B, T]，有效位置为 True（或 1），无效位置为 False（或 0）
        """
        w = self.project(z)  # [B, T, 1]
        w = w.squeeze(-1)    # [B, T]

        if mask is not None:
            w = w.masked_fill(~mask, float('-inf'))  # 屏蔽无效位置

        beta = torch.softmax(w, dim=1)  # [B, T]
        beta = beta.unsqueeze(-1)       # [B, T, 1]
        return (beta * z).sum(dim=1)    # [B, D]


class ModeProduct(nn.Module):
    def __init__(self, i, j, k, a, b, c):
        super(ModeProduct, self).__init__()
        self.A = nn.Parameter(torch.rand(i, a))
        self.B = nn.Parameter(torch.rand(j, b))
        self.C = nn.Parameter(torch.rand(k, c))

        self.b = nn.Parameter(torch.rand(a, b, c))

    def forward(self, x):
        ret = torch.sigmoid(
            torch.einsum('nijk,kc->nijc', torch.einsum('nijk,jc->nick', torch.einsum('nijk,ic->ncjk', x, self.A), self.B), self.C) + self.b)
        return ret
