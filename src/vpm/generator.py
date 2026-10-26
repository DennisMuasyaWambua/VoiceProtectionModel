"""Causal streaming perturbation generator G_θ (§3.6.2, Table 3.5).

* 16 kHz audio, 10 ms hops (H=160). The analysis window for block b is the 20 ms ending at the
  end of block b, so windowing adds no look-ahead.
* Per-block features: learned projection of the raw 320-sample window + 40-band log-mel.
* Backbone: dilated depth-wise separable causal TCN (left-padded), optional GRU bottleneck.
* Output: 160 perturbation samples per block, ε·tanh(·)  (R3: budget by construction).
* Look-ahead L blocks implemented as a fixed output delay: δ for block b is produced once
  block b+L has arrived (R1). L=0 → G-S, L=1 (10 ms) → G-M.
* `forward` (offline) and `StreamingGenerator.step` (block-by-block with fixed-size state, R2)
  are numerically identical — see tests/test_streaming.py.
"""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
import torchaudio

H, WIN, NMEL, NFFT = 160, 320, 40, 512
MODES = ["mobile", "pa_live", "unknown"]

def _mel_fb():
    return torchaudio.functional.melscale_fbanks(NFFT // 2 + 1, 20.0, 8000.0, NMEL, 16000, norm="slaney", mel_scale="slaney")

class FrameFeatures(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.register_buffer("win", torch.hann_window(WIN))
        self.register_buffer("fb", _mel_fb())
        # real DFT as matmul (ONNX/INT8-friendly; numerically equal to rfft(n=512) of the 320-sample frame)
        n = torch.arange(WIN, dtype=torch.float64)[:, None]; k = torch.arange(NFFT // 2 + 1, dtype=torch.float64)[None]
        ang = 2 * math.pi * n * k / NFFT
        self.register_buffer("dft_cos", torch.cos(ang).float(), persistent=False)
        self.register_buffer("dft_sin", torch.sin(ang).float(), persistent=False)
        self.wav_proj = nn.Linear(WIN, C)
        self.mel_proj = nn.Linear(NMEL, C)
        self.mode_emb = nn.Linear(len(MODES), C, bias=False)
    def forward(self, frames, mode):  # frames (B,T,320), mode (B,3)
        fw = frames * self.win
        spec = (fw @ self.dft_cos) ** 2 + (fw @ self.dft_sin) ** 2
        mel = torch.log(spec @ self.fb + 1e-6)
        return self.wav_proj(frames * 10.0) + self.mel_proj(mel / 10.0) + self.mode_emb(mode).unsqueeze(1)

class TCNBlock(nn.Module):
    def __init__(self, C, d, k=3):
        super().__init__()
        self.d, self.k = d, k
        self.dw = nn.Conv1d(C, C, k, dilation=d, groups=C)
        self.act = nn.PReLU(C)
        self.pw = nn.Conv1d(C, C, 1)
        self.norm = nn.LayerNorm(C)
    @property
    def pad(self): return (self.k - 1) * self.d
    def core(self, xin_padded, res):  # xin_padded (B,C,T+pad)
        y = self.pw(self.act(self.dw(xin_padded)))
        return res + self.norm(y.transpose(1, 2)).transpose(1, 2)
    def forward(self, x):  # (B,C,T)
        return self.core(F.pad(x, (self.pad, 0)), x)

class Generator(nn.Module):
    def __init__(self, C=64, stacks=2, blocks=5, gru=False, lookahead=0, eps=0.008):
        super().__init__()
        self.cfg = dict(C=C, stacks=stacks, blocks=blocks, gru=gru, lookahead=lookahead, eps=eps)
        self.eps, self.L = eps, lookahead
        self.feat = FrameFeatures(C)
        self.tcn = nn.ModuleList([TCNBlock(C, 2 ** b) for _ in range(stacks) for b in range(blocks)])
        self.gru = nn.GRU(C, C, batch_first=True) if gru else None
        self.out = nn.Linear(C, H)
        nn.init.normal_(self.out.weight, std=0.01); nn.init.zeros_(self.out.bias)  # start near-silent δ

    @staticmethod
    def frames(x):  # x (B,N) with N % H == 0 → (B,T,320) windows ending at each block end
        B = x.shape[0]
        prev = F.pad(x, (H, 0))[:, : x.shape[1]].reshape(B, -1, H)
        return torch.cat([prev, x.reshape(B, -1, H)], -1)

    def forward(self, x, mode=None):
        """x: (B,N) clean waveform, returns δ (B,N)."""
        B, N = x.shape
        assert N % H == 0, "length must be a multiple of 160"
        if mode is None: mode = torch.zeros(B, 3, device=x.device); mode[:, 2] = 1
        xin = F.pad(x, (0, self.L * H))  # future blocks for look-ahead (zero beyond end)
        h = self.feat(self.frames(xin), mode).transpose(1, 2)
        for blk in self.tcn: h = blk(h)
        h = h.transpose(1, 2)
        if self.gru is not None: h = h + self.gru(h)[0]
        d = self.eps * torch.tanh(self.out(h))  # (B,T+L,160)
        d = d[:, self.L:, :]  # δ for block b comes from output step b+L
        return d.reshape(B, -1)

    def n_params(self): return sum(p.numel() for p in self.parameters())

    def macs_per_second(self):
        C = self.cfg["C"]; per_frame = WIN * C + NMEL * C + (NFFT // 2 + 1) * NMEL + NFFT * 9 * 2  # proj + mel + ~FFT
        per_frame += len(self.tcn) * (3 * C + C * C)
        if self.gru is not None: per_frame += 6 * C * C
        per_frame += C * H
        return per_frame * 100

class StreamingGenerator(nn.Module):
    """Block-by-block wrapper with explicit fixed-size state (exported to ONNX).
    Inputs: block (1,160) newest audio block, prev (1,160) previous block, mode (1,3), conv state
    (1,C,S) packing all TCN left-contexts, gru state (1,1,C). Output: δ for block (t−L), new states."""
    def __init__(self, g: Generator):
        super().__init__()
        self.g = g
        self.pads = [b.pad for b in g.tcn]
        self.S = sum(self.pads)
        self.C = g.cfg["C"]
    def init_state(self):
        return (torch.zeros(1, H), torch.zeros(1, self.C, self.S), torch.zeros(1, 1, self.C))
    def forward(self, block, prev, mode, conv_state, gru_state):
        frame = torch.cat([prev, block], -1).unsqueeze(1)  # (1,1,320)
        h = self.g.feat(frame, mode).transpose(1, 2)  # (1,C,1)
        new_states, o = [], 0
        for blk, p in zip(self.g.tcn, self.pads):
            st = conv_state[:, :, o:o + p]; o += p
            xp = torch.cat([st, h], -1)
            new_states.append(xp[:, :, 1:])
            h = blk.core(xp, h)
        h = h.transpose(1, 2)
        if self.g.gru is not None:
            y, gru_state = self.g.gru(h, gru_state.transpose(0, 1).contiguous())
            h = h + y; gru_state = gru_state.transpose(0, 1)
        d = self.g.eps * torch.tanh(self.g.out(h)).reshape(1, H)
        return d, block, torch.cat(new_states, -1), gru_state

def G_S(eps=0.008): return Generator(C=64, stacks=2, blocks=5, gru=False, lookahead=0, eps=eps)
def G_M(eps=0.008): return Generator(C=96, stacks=3, blocks=6, gru=True, lookahead=1, eps=eps)

def stream_process(sg: StreamingGenerator, x: torch.Tensor, mode=None):
    """Run the streaming model over a waveform; returns δ aligned to x (look-ahead compensated)."""
    L = sg.g.L
    if mode is None: mode = torch.tensor([[0., 0., 1.]])
    prev, cs, gs = sg.init_state()
    xin = F.pad(x.view(1, -1), (0, L * H)); outs = []
    with torch.no_grad():
        for b in range(xin.shape[-1] // H):
            d, prev, cs, gs = sg(xin[:, b * H:(b + 1) * H], prev, mode, cs, gs)
            outs.append(d)
    return torch.cat(outs[L:], -1).view(-1)
