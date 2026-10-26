"""E2 adaptive white-box attacker: a removal network trained on (protected, clean) pairs produced by the
released generator (Table 3.1). Research use only — NOT to be released (§3.14 dual-use policy)."""
import random, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .losses import mr_stft_loss

class RemovalNet(nn.Module):
    def __init__(self, C=64):
        super().__init__()
        L = []; c = 1
        for d in [1, 2, 4, 8, 16, 32, 1]:
            L += [nn.Conv1d(c, C, 7, padding=3 * d, dilation=d), nn.PReLU(C)]; c = C
        self.net = nn.Sequential(*L, nn.Conv1d(C, 1, 7, padding=3))
    def forward(self, y): return y - 0.02 * torch.tanh(self.net(y[:, None]))[:, 0]  # predicts the residual

def train_removal(gen, train_paths, steps=600, batch=8, N=32000, seed=0, log=print):
    from .train import load_crop
    torch.manual_seed(seed); rng = random.Random(seed); net = RemovalNet(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    gen.eval(); hist = []
    for s in range(steps):
        x = torch.tensor(np.stack([load_crop(p, N, rng) for p in rng.sample(train_paths, batch)]))
        with torch.no_grad(): xp = (x + gen(x)).clamp(-1, 1)
        loss = F.l1_loss(net(xp), x) * 100 + mr_stft_loss(x, net(xp))
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(loss.item())
        if s % 100 == 0: log(f"removal step {s} loss {loss.item():.4f}")
    return net, hist

def load_removal(path):
    net = RemovalNet(); net.load_state_dict(torch.load(path)); net.eval()
    return lambda x: net(x[None]).detach()[0]
