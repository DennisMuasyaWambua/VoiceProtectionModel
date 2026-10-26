"""Dataset discovery, cleaning (§3.3.5) and speaker-disjoint splits."""
from __future__ import annotations
import glob, os, random
import numpy as np, pandas as pd, soundfile as sf, librosa, pyloudnorm as pyln

SR = 16000

def list_librispeech(root: str, subset: str) -> pd.DataFrame:
    rows = []
    spk_info = {}
    sp = os.path.join(root, "LibriSpeech", "SPEAKERS.TXT")
    if os.path.exists(sp):
        for line in open(sp, encoding="utf-8", errors="ignore"):
            if line.startswith(";"): continue
            p = [s.strip() for s in line.split("|")]
            if len(p) >= 3: spk_info[p[0]] = p[1]
    for f in sorted(glob.glob(os.path.join(root, "LibriSpeech", subset, "*", "*", "*.flac"))):
        spk, ch, utt = os.path.basename(f)[:-5].split("-")
        rows.append(dict(corpus="librispeech", subset=subset, path=f, speaker=f"ls{spk}",
                         utt_id=os.path.basename(f)[:-5], gender=spk_info.get(spk, "?"), lang="en"))
    # transcripts
    trans = {}
    for t in glob.glob(os.path.join(root, "LibriSpeech", subset, "*", "*", "*.trans.txt")):
        for line in open(t):
            k, v = line.strip().split(" ", 1); trans[k] = v
    df = pd.DataFrame(rows); df["text"] = df.utt_id.map(trans)
    return df

def list_fleurs(root: str, split: str) -> pd.DataFrame:
    tsv = os.path.join(root, "fleurs_sw_ke", f"{split}.tsv")
    cols = ["id", "file", "raw_text", "text", "phon", "num_samples", "gender"]
    d = pd.read_csv(tsv, sep="\t", header=None, names=cols, quoting=3)
    d["path"] = d.file.map(lambda f: os.path.join(root, "fleurs_sw_ke", split, f))
    d = d[d.path.map(os.path.exists)]
    # FLEURS has no speaker IDs; the sentence id groups recordings of the same prompt.
    return pd.DataFrame(dict(corpus="fleurs_sw_ke", subset=split, path=d.path, speaker="unknown",
                             utt_id=d.file.str.replace(".wav", "", regex=False), gender=d.gender.str[0].str.upper(),
                             lang="sw", text=d.text))

_meter = pyln.Meter(SR)

def clean_one(path: str, cfg: dict):
    """Load → mono 16 kHz → VAD trim → QC → −26 LUFS. Returns (audio, info) or (None, reason)."""
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        return None, f"corrupt:{type(e).__name__}"
    x = x.mean(1)
    if sr != SR: x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    if not np.isfinite(x).all(): return None, "nonfinite"
    clip_frac = float(np.mean(np.abs(x) > 0.999))
    if clip_frac > cfg["clip_frac_max"]: return None, "clipping"
    y, _ = librosa.effects.trim(x, top_db=cfg["trim_top_db"])
    dur = len(y) / SR
    if dur < cfg["min_dur"]: return None, "too_short"
    if dur > cfg["max_dur"]: y = y[: int(cfg["max_dur"] * SR)]
    try: lufs_in = float(_meter.integrated_loudness(y))
    except Exception: return None, "loudness_fail"
    if not np.isfinite(lufs_in): return None, "silent"
    y = pyln.normalize.loudness(y, lufs_in, cfg["target_lufs"]).astype(np.float32)
    peak = float(np.abs(y).max())
    if peak > 0.99: y = y * (0.99 / peak)  # peak guard (rare)
    # crude SNR estimate: 95th vs 10th percentile frame energy
    fr = librosa.util.frame(y, frame_length=400, hop_length=160)
    e = 10 * np.log10((fr ** 2).mean(0) + 1e-10)
    snr = float(np.percentile(e, 95) - np.percentile(e, 10))
    return y, dict(dur=len(y) / SR, raw_dur=len(x) / SR, lufs_in=lufs_in, snr_est=snr, clip_frac=clip_frac)

def speaker_split(df: pd.DataFrame, n_train: int, n_val: int, seed: int = 0) -> pd.DataFrame:
    """Gender-balanced speaker-disjoint train/val split."""
    rng = random.Random(seed); out = {}
    for g, sub in df.groupby("gender"):
        spk = sorted(sub.speaker.unique()); rng.shuffle(spk)
        k = round(n_train * len(spk) / df.speaker.nunique())
        for s in spk[:k]: out[s] = "train"
        for s in spk[k:]: out[s] = "val"
    return df.assign(split=df.speaker.map(out))
