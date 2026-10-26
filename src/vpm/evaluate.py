"""Evaluation: verifier acceptance of protected references (proxy for clone acceptance), ΔAR,
SIM, PESQ/STOI/SNR, and ΔWER (H5) — with speaker-cluster bootstrap CIs."""
from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F
from pesq import pesq
from pystoi import stoi

def eer_threshold(gen, imp):
    s = np.sort(np.concatenate([gen, imp])); best = (1, None)
    for t in s:
        far = (imp >= t).mean(); frr = (gen < t).mean()
        if abs(far - frr) < best[0]: best = (abs(far - frr), t, (far + frr) / 2)
    return float(best[1]), float(best[2])

def quality(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    out = {"snr_db": float(10 * np.log10((x ** 2).sum() / (((y - x) ** 2).sum() + 1e-12)))}
    try: out["pesq_wb"] = float(pesq(16000, x, y, "wb"))
    except Exception: out["pesq_wb"] = np.nan
    out["stoi"] = float(stoi(x, y, 16000, extended=False))
    out["linf"] = float(np.abs(y - x).max())
    return out

def cluster_bootstrap(df, value, cluster="speaker", n=2000, seed=0, stat=np.mean):
    rng = np.random.default_rng(seed); groups = {k: v[value].values for k, v in df.groupby(cluster)}
    keys = list(groups); res = []
    for _ in range(n):
        s = rng.choice(keys, len(keys), replace=True)
        res.append(stat(np.concatenate([groups[k] for k in s])))
    return float(stat(df[value].values)), float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))
