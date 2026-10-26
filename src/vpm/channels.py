"""Kenyan transformation distribution T_KE (Table 3.3), scaled-down, and held-out channels (Table 3.4).

Differentiable ops (resample, band-pass, convolution, mixing, noise, clipping) have exact gradients.
Real codecs (Opus/MP3 via ffmpeg) are non-differentiable and use identity-STE BPDA
(forward = real codec, backward = identity; ablation A1 'identity-STE' variant). AMR-NB/WB
encoders are not available in this ffmpeg build, so the narrowband family uses
8 kHz resampling + 300–3400 Hz band-pass + G.711 mu-law (STE) as a proxy.
"""
from __future__ import annotations
import math, os, random, subprocess, tempfile
import numpy as np, torch, torch.nn.functional as F, torchaudio.functional as AF, soundfile as sf

SR = 16000

def _fft_bandpass(x, lo, hi, sr=SR):
    X = torch.fft.rfft(x, dim=-1)
    f = torch.fft.rfftfreq(x.shape[-1], 1 / sr).to(x.device)
    m = ((f >= lo) & (f <= hi)).float()
    return torch.fft.irfft(X * m, n=x.shape[-1], dim=-1)

def _ste(x, y):
    """forward y, backward identity."""
    return x + (y - x).detach()

def mulaw(x, mu=255):
    y = torch.sign(x) * torch.log1p(mu * x.abs().clamp(max=1)) / math.log1p(mu)
    y = torch.round((y + 1) / 2 * mu) / mu * 2 - 1
    y = torch.sign(y) * ((1 + mu) ** y.abs() - 1) / mu
    return _ste(x, y)

def narrowband(x, mu=True):
    y = AF.resample(x, SR, 8000); y = AF.resample(y, 8000, SR)[..., : x.shape[-1]]
    y = _fft_bandpass(y, 300, 3400)
    return mulaw(y) if mu else y

def ffmpeg_codec(x_np: np.ndarray, codec: str, bitrate: str, sr_codec: int = 16000) -> np.ndarray:
    """Real encode→decode through ffmpeg. codec in {'libopus','libmp3lame'}."""
    ext = {"libopus": "ogg", "libmp3lame": "mp3"}[codec]
    with tempfile.TemporaryDirectory() as d:
        i, o, r = f"{d}/i.wav", f"{d}/o.{ext}", f"{d}/r.wav"
        sf.write(i, np.clip(x_np, -1, 1), SR, subtype="PCM_16")
        extra = ["-application", "voip"] if codec == "libopus" else []
        osr = 48000 if codec == "libopus" else sr_codec
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", i, "-c:a", codec, "-b:a", bitrate, *extra,
                        "-ar", str(osr), o], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", o, "-ar", str(SR), "-ac", "1", r], check=True)
        y, _ = sf.read(r, dtype="float32")
    n = len(x_np)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)

def codec_ste(x: torch.Tensor, codec: str, bitrate: str) -> torch.Tensor:
    y = torch.stack([torch.from_numpy(ffmpeg_codec(xi.detach().cpu().numpy(), codec, bitrate)) for xi in x]).to(x)
    return _ste(x, y)

def synth_rir(rt60: float, sr=SR, length_s=None, gen=None):
    L = int(sr * (length_s or min(rt60, 1.0)))
    t = torch.arange(L) / sr
    h = torch.randn(L, generator=gen) * torch.exp(-6.9078 * t / rt60)
    h[0] = 1.0  # direct sound from loudspeaker
    return h / h.norm()

def fft_conv(x, h):
    n = x.shape[-1] + h.shape[-1] - 1
    y = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(h.to(x), n), n)
    return y[..., : x.shape[-1]]

def add_noise(x, snr_db):
    n = torch.randn_like(x)
    # pink-ish (Nairobi ambience/crowd proxy): integrate white noise lightly
    n = n + 0.9 * torch.roll(n, 1, -1)
    p_x = x.pow(2).mean(-1, keepdim=True) + 1e-8
    p_n = n.pow(2).mean(-1, keepdim=True)
    return x + n * torch.sqrt(p_x / (p_n * 10 ** (snr_db / 10)))

def pa_rerecord(x, x_clean, rng: random.Random):
    """PA/re-recording family: loudspeaker band-limit, RIR (RT60 0.4–1.6 s), distance attenuation,
    *unprotected direct path* at direct-to-PA ratio −20..+6 dB, crowd noise SNR 5–25 dB, soft clipping."""
    y = _fft_bandpass(x, 120, 7000)  # PA loudspeaker response proxy
    h = synth_rir(rng.uniform(0.4, 1.6))
    y = fft_conv(y, h)
    dist = rng.uniform(1, 15); y = y / dist
    dpr = rng.uniform(-20, 6)
    d = fft_conv(x_clean, synth_rir(rng.uniform(0.4, 1.6)))
    d = d * (y.norm(dim=-1, keepdim=True) / (d.norm(dim=-1, keepdim=True) + 1e-8)) * 10 ** (dpr / 20)
    y = y + d.detach()
    y = add_noise(y, rng.uniform(5, 25))
    c = rng.uniform(0.7, 1.0)
    y = y / (y.abs().amax(-1, keepdim=True) + 1e-8) * 0.9
    return torch.tanh(y / c) * c

def device(x, rng: random.Random):
    """Always applied: loudness/AGC change (±), additive ambience SNR 10–30 dB, time shift 0–20 ms."""
    g = 10 ** (rng.uniform(-4, 4) / 20)
    y = add_noise(x * g, rng.uniform(10, 30))
    s = rng.randint(0, int(0.02 * SR))
    return F.pad(y, (s, 0))[..., : x.shape[-1]]

FAMILIES = [("narrowband", 0.22), ("wideband", 0.15), ("voicenote", 0.25), ("pa", 0.23), ("clean", 0.15)]

def sample_ke(x, x_clean, rng: random.Random, use_real_codecs=True, families=None):
    """Sample one transformation t ~ T_KE and apply it. Returns (y, description)."""
    fams = families or FAMILIES
    names, ps = zip(*fams)
    hops = rng.choice([2, 3]) if rng.random() < 0.15 else 1
    y, desc = x, []
    for _ in range(hops):
        fam = rng.choices(names, ps)[0]; desc.append(fam)
        if fam == "narrowband": y = narrowband(y)
        elif fam == "wideband":  # AMR-WB proxy: 50–7000 Hz + mild quantisation noise
            y = _fft_bandpass(y, 50, 7000); y = _ste(y, torch.round(y * 4096) / 4096)
        elif fam == "voicenote":
            br = rng.choice(["8k", "12k", "16k", "24k", "32k"])
            y = codec_ste(y, "libopus", br) if use_real_codecs else _fft_bandpass(y, 0, 8000); desc[-1] += f"@{br}"
        elif fam == "pa": y = pa_rerecord(y, x_clean, rng)
    y = device(y, rng)
    return y, "+".join(desc)

# ---------- Held-out channels (never used in training) ----------
def heldout_channels():
    return {
        "none": lambda x: x,
        "g711_mulaw_8k": lambda x: narrowband(x, mu=True),
        "opus_6k": lambda x: codec_ste(x, "libopus", "6k"),
        "opus_48k": lambda x: codec_ste(x, "libopus", "48k"),
        "mp3_32k": lambda x: codec_ste(x, "libmp3lame", "32k"),
        "chain_opus12k_g711": lambda x: narrowband(codec_ste(x, "libopus", "12k")),
    }

# ---------- Generic removal attacks (E4-style, grey-box) ----------
def removal_attacks():
    def lowpass_resample(x):
        y = AF.resample(x, SR, 11025); y = AF.resample(y, 11025, SR)[..., : x.shape[-1]]
        return _fft_bandpass(y, 80, 4000)
    def spectral_gate(x):
        S = torch.stft(x, 512, 128, window=torch.hann_window(512), return_complex=True)
        mag = S.abs(); floor = torch.quantile(mag, 0.2, dim=-1, keepdim=True)
        mask = (mag > 2 * floor).float()
        return torch.istft(S * mask, 512, 128, window=torch.hann_window(512), length=x.shape[-1])
    return {"bandpass_mp3_64k": lambda x: codec_ste(_fft_bandpass(x, 100, 7000), "libmp3lame", "64k"),
            "lowpass_resample_11k": lowpass_resample,
            "spectral_gate_denoise": spectral_gate}
