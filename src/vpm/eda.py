"""EDA plotting helpers."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, soundfile as sf, librosa, librosa.display

def summary_table(m):
    return m.groupby(["corpus", "split"]).agg(utterances=("utt_id", "size"), speakers=("speaker", "nunique"),
        hours=("dur", lambda d: round(d.sum() / 3600, 2)), mean_dur_s=("dur", "mean"), median_snr_est_db=("snr_est", "median"),
        female_share=("gender", lambda g: round((g == "F").mean(), 2))).round(2)

def plot_overview(m, ex, path):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for (c, s), g in m.groupby(["corpus", "split"]): ax[0, 0].hist(g.dur, bins=40, alpha=.5, label=f"{c}/{s}")
    ax[0, 0].set(title="Duration after trimming (s)", xlabel="s"); ax[0, 0].legend(fontsize=7)
    ls = m[m.corpus == "librispeech"]
    per = ls.groupby(["split", "speaker"]).dur.sum().div(60).reset_index()
    for s, g in per.groupby("split"): ax[0, 1].hist(g.dur, bins=15, alpha=.6, label=s)
    ax[0, 1].set(title="LibriSpeech minutes per speaker", xlabel="min"); ax[0, 1].legend()
    gc = m.groupby(["corpus", "split", "gender"]).size().unstack(fill_value=0)
    gc.plot.bar(ax=ax[0, 2], stacked=True); ax[0, 2].set(title="Gender (speaker-level for LS; clip-level for FLEURS)"); ax[0, 2].tick_params(axis="x", rotation=30, labelsize=7)
    for c, g in m.groupby("corpus"): ax[1, 0].hist(g.lufs_in, bins=40, alpha=.5, label=c)
    ax[1, 0].set(title="Input loudness before normalisation (LUFS) → all set to −26", xlabel="LUFS"); ax[1, 0].legend()
    for c, g in m.groupby("corpus"): ax[1, 1].hist(g.snr_est, bins=40, alpha=.5, label=c)
    ax[1, 1].set(title="Estimated SNR (95th−10th pct frame energy, dB)"); ax[1, 1].legend()
    ex.status.value_counts().plot.bar(ax=ax[1, 2]); ax[1, 2].set(title=f"Excluded clips (n={len(ex)})")
    fig.tight_layout(); fig.savefig(path, dpi=110); return fig

def plot_spectrograms(paths, titles, path):
    fig, ax = plt.subplots(1, len(paths), figsize=(5 * len(paths), 3.5))
    for a, p, t in zip(np.atleast_1d(ax), paths, titles):
        y, _ = sf.read(p); S = librosa.amplitude_to_db(np.abs(librosa.stft(y[:16000 * 4], n_fft=512, hop_length=160)), ref=np.max)
        librosa.display.specshow(S, sr=16000, hop_length=160, x_axis="time", y_axis="hz", ax=a); a.set_title(t, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); return fig
