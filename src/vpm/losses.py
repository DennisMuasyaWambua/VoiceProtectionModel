"""Perceptual losses (Eqs. 5–7): multi-resolution STFT + psychoacoustic masking hinge."""
import torch, math

RES = [(512, 50, 240), (1024, 120, 600), (2048, 240, 1200)]  # (fft, hop, win) as in §3.5.3

def _stft(x, n, h, w):
    return torch.stft(x, n, h, w, window=torch.hann_window(w, device=x.device), return_complex=True).abs() + 1e-7

def mr_stft_loss(x, y):
    loss = 0.0
    for n, h, w in RES:
        X, Y = _stft(x, n, h, w), _stft(y, n, h, w)
        sc = torch.linalg.norm(Y - X, dim=(1, 2)) / torch.linalg.norm(X, dim=(1, 2))
        mag = (Y.log() - X.log()).abs().mean(dim=(1, 2))
        loss = loss + (sc + mag).mean()
    return loss / len(RES)

def _spread_db(p_db):
    # simple Schroeder-like spreading across frequency bins (triangular kernel), in dB domain approximated
    k = torch.tensor([0.05, 0.15, 0.6, 1.0, 0.6, 0.15, 0.05], device=p_db.device)
    lin = 10 ** (p_db / 10)
    B, Fq, T = lin.shape
    s = torch.nn.functional.conv1d(lin.permute(0, 2, 1).reshape(B * T, 1, Fq), k.view(1, 1, -1), padding=3)
    return 10 * torch.log10(s.view(B, T, Fq).permute(0, 2, 1) + 1e-12)

def masking_loss(x, delta, offset_db=12.0, n_fft=512, hop=160):
    """Eq. 6: hinge of normalised perturbation PSD above a frequency-masking threshold of x.
    θ_x = spread PSD of x − offset (simplified Qin et al. 2019 threshold)."""
    win = torch.hann_window(n_fft, device=x.device)
    Px = torch.stft(x, n_fft, hop, window=win, return_complex=True).abs() ** 2
    Pd = torch.stft(delta, n_fft, hop, window=win, return_complex=True).abs() ** 2
    ref = Px.amax(dim=(1, 2), keepdim=True) + 1e-12  # common normalisation (96 dB SPL ≙ max)
    px_db = 96 + 10 * torch.log10(Px / ref + 1e-12)
    pd_db = 96 + 10 * torch.log10(Pd / ref + 1e-12)
    theta = _spread_db(px_db) - offset_db
    return torch.relu(pd_db - theta).mean() / 10.0

def perceptual_loss(x, y, gamma=1.0):
    return mr_stft_loss(x, y) + gamma * masking_loss(x, y - x)

def snr_db(x, d):
    return 10 * torch.log10(x.pow(2).sum(-1) / (d.pow(2).sum(-1) + 1e-12))
