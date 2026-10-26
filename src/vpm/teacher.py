"""Per-utterance PGD-EOT teacher (Eqs. 3, 8, 9) — baseline B4 and ε trade-off curve."""
import random, torch, torch.nn.functional as F
from .losses import perceptual_loss
from . import channels as ch

def pgd_eot(x, enc, eps=0.008, K=100, M=2, lam=0.1, gamma=1.0, seed=0, use_real_codecs=False):
    """x (1,N). Untargeted: minimise E_t cos(E(t(x+δ)), E(x)) + λ L_perc, ||δ||∞≤ε, sign-PGD, α=2.5ε/K."""
    g = torch.Generator().manual_seed(seed); rng = random.Random(seed)
    with torch.no_grad(): e0 = enc(x)
    d = (torch.rand(x.shape, generator=g) * 2 - 1) * eps
    alpha = 2.5 * eps / K
    best, best_l = d.clone(), 1e9
    for k in range(K):
        d.requires_grad_(True); xp = (x + d).clamp(-1, 1)
        ys = torch.cat([ch.sample_ke(xp, x, rng, use_real_codecs=use_real_codecs)[0] for _ in range(M)])
        l_adv = F.cosine_similarity(enc(ys), e0.expand(M, -1)).mean()
        loss = l_adv + lam * perceptual_loss(x, xp, gamma)
        gr, = torch.autograd.grad(loss, d)
        with torch.no_grad():
            if loss.item() < best_l: best_l, best = loss.item(), d.detach().clone()
            d = (d - alpha * gr.sign()).clamp(-eps, eps)
            d = (x + d).clamp(-1, 1) - x
    return best.detach()
