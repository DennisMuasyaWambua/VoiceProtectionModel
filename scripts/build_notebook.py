import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))
md("""# Voice Protection Model — streaming adversarial perturbations against voice cloning
**Dennis — MSc Data Science & Analytics, Strathmore University.** Research prototype implementing Chapter 3 (Research Methodology) of the proposal at CPU scale.

Pipeline (CRISP-DM, §3.1): **data → cleaning → EDA → preprocessing (10 ms blocks / look-ahead) → training of the causal streaming generator G-M (B6) and baselines → evaluation under the threat model (Table 3.1) → ONNX export → INT8 optimisation & latency benchmark.**

> All numbers in this notebook are computed from real runs on this machine (8 vCPU, no GPU, ~15 GB RAM). Long jobs (generator training, full evaluation) were run with the scripts in `scripts/` and their outputs are loaded here; set `RERUN=True` to recompute. **Everything scaled down from the proposal is listed in §11.**""")
code("""import os, sys, json, glob, warnings; warnings.filterwarnings('ignore')
ROOT = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
os.chdir(ROOT); sys.path.insert(0, 'src')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, yaml, soundfile as sf
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
cfg = yaml.safe_load(open('config.yaml')); RERUN = False
print('torch', torch.__version__, '| threads', torch.get_num_threads(), '| CUDA', torch.cuda.is_available())""")
md("""## 1. Datasets — sources, licences, rationale
| Corpus | Licence | Access | Used for | Why |
|---|---|---|---|---|
| **LibriSpeech dev-clean** (OpenSLR SLR12) | CC BY 4.0 | direct download, no login | generator **training** (30 speakers) + **validation**/λ selection (10 speakers) | Proposal's training/clone corpus (Table 3.2); cloning is restricted to LibriSpeech. *train-clean-100 (251 spk, 6.3 GB) swapped for dev-clean for CPU budget.* |
| **LibriSpeech test-clean** | CC BY 4.0 | direct | **test** (40 speakers, disjoint) | Exactly the proposal's 40 English test speakers |
| **FLEURS Swahili – Kenya (`sw_ke`)** (Google) | CC BY 4.0 | Hugging Face, no login | Kenyan-locale Swahili **utility & embedding analysis only** (no cloning) | Stand-in for Common Voice Swahili / NBVS (Kenyan Swahili) |

Not used: **Common Voice** (now gated behind Mozilla Data Collective sign-in/terms), **VoxCeleb** (requires request form), **AfriSpeech-200** (CC BY-NC-SA, licence for this use 'to be confirmed' in the proposal), **ALFFA** (ASR sanity checks only in the proposal). NBVS, the real Safaricom/Airtel channel set and the lecture-hall PA set are new data to be collected and do not exist yet. Re-download with `scripts/download_data.sh`.""")
md("## 2. Cleaning (§3.3.5)\nmono 16 kHz → energy-VAD trim (top_db 30) → drop corrupt / clipped (>0.1 % samples at full scale) / <2 s → cap 20 s → **−26 LUFS** (so ε means the same everywhere) → speaker-disjoint, gender-balanced splits. Run by `scripts/prepare_data.py` (parallel, 8 workers).")
code("""if RERUN: os.system('PYTHONPATH=src python scripts/prepare_data.py')
m = pd.read_csv('data/manifests/all.csv'); ex = pd.read_csv('data/manifests/excluded.csv')
from vpm.eda import summary_table, plot_overview, plot_spectrograms
print('kept', len(m), '| excluded', len(ex)); print(ex.groupby(['corpus','status']).size())
# speaker-disjointness check
sp = m[m.corpus=='librispeech'].groupby('speaker').split.nunique(); assert (sp==1).all(); print('speaker-disjoint splits: OK')
summary_table(m)""")
md("## 3. Exploratory data analysis")
code("""fig = plot_overview(m, ex, 'plots/eda_overview.png'); plt.show()""")
code("""ls = m[m.corpus=='librispeech']
print(ls.groupby(['split','gender']).speaker.nunique().unstack())
print(m.groupby('corpus')[['dur','raw_dur','lufs_in','snr_est']].describe().T.round(2))""")
code("""ex1 = m[(m.split=='train')].iloc[0]; ex2 = m[m.split=='sw_test'].iloc[0]
fig = plot_spectrograms([ex1.clean_path, ex2.clean_path], [f'LibriSpeech {ex1.utt_id} ({ex1.gender})', f'FLEURS sw_ke {ex2.utt_id} ({ex2.gender})'], 'plots/eda_spectrograms.png'); plt.show()""")
md("""## 4. Preprocessing for the streaming design (§3.6.2)
16 kHz audio in **10 ms hops (160 samples)**; each block's **20 ms analysis window ends at the current block** (previous + current block), so windowing adds no look-ahead. Per-block features = learned projection of the raw 320-sample window + **40-band log-mel** (DFT implemented as a matmul so it exports to ONNX/INT8). Look-ahead L is a fixed output delay: G-M uses **L = 1 block = 10 ms** (algorithmic latency 20 ms), G-S uses L = 0 (10 ms).""")
code("""from vpm.generator import G_S, G_M, Generator, StreamingGenerator, stream_process
x = torch.tensor(sf.read(ex1.clean_path, dtype='float32')[0][:16000]); fr = Generator.frames(x[None])
print('waveform', tuple(x.shape), '→ blocks/windows', tuple(fr.shape), '(T blocks × 320-sample window)')
for g in (G_S(), G_M()):
    print(f"{'G-S' if g.L==0 else 'G-M'}: params={g.n_params():,}  ~MAC/s={g.macs_per_second()/1e6:.1f} M  look-ahead={g.L*10} ms  algorithmic latency={10+10*g.L} ms")
# unit test (§3.6.2): streaming == offline
g = G_M().eval(); xx = torch.randn(1, 160*80)*0.05
with torch.no_grad(): off = g(xx)[0]
print('max |offline − streaming| =', (off - stream_process(StreamingGenerator(g).eval(), xx[0])).abs().max().item())""")
code("""import random, librosa, librosa.display
from vpm import channels as ch
xc = x[None]; ex = [('clean', xc)] + [(d, y) for y, d in (ch.sample_ke(xc, xc, random.Random(s)) for s in (1, 4, 7))]
fig, ax = plt.subplots(1, 4, figsize=(18, 3))
for a, (name, y) in zip(ax, ex):
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y[0].detach().numpy(), n_fft=512, hop_length=160)), ref=np.max)
    librosa.display.specshow(S, sr=16000, hop_length=160, ax=a, y_axis='hz'); a.set_title(name)
plt.tight_layout(); plt.savefig('plots/channel_examples.png', dpi=100); plt.show()""")
md("""## 5. Training (Eq. 11; §3.6.4)
Objective: \\(\\mathbb{E}_x\\mathbb{E}_{t\\sim T_{KE}}\\cos(E(t(x+\\delta_\\theta)),E(x)) + \\lambda(\\mathcal L_{stft}+\\gamma\\mathcal L_{mask})\\), seen encoder **ECAPA-TDNN** (SpeechBrain, VoxCeleb), ε = 0.008 (−42 dBFS, mid-grid), AdamW 3e-4 cosine, grad-clip 1, curriculum (clean+device → codecs → full incl. PA/re-recording & chains), mode switch (mobile / PA-live / unknown). Real Opus via ffmpeg with identity-STE BPDA; AMR replaced by a narrowband (8 kHz, 300–3400 Hz, G.711 µ-law) proxy.

Runs (`python -m vpm.train ...`): **B6 G-M T_KE λ=0.1** (main), G-M T_KE λ=1 (λ-grid point, stopped at 250 steps: no protection), **B5 G-M T_gen λ=0.1** (generic channel baseline for H2), **G-S T_KE λ=0.1** (0 ms look-ahead, A7/A8).""")
code("""runs = {p.split('/')[1]: pd.read_csv(p) for p in sorted(glob.glob('models/*/log.csv'))}
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for k, d in runs.items():
    v = d.dropna(subset=['val_cos'])
    ax[0].plot(d.step, d.loss.rolling(20).mean(), label=k); ax[1].plot(v.step, v.val_cos, 'o-', label=k)
    ax[2].plot(v.step, v.val_cos_nb, 'o-', label=k); ax[3].plot(v.step, v.val_snr, 'o-', label=k)
for a, t in zip(ax, ['train loss (rolling 20)', 'val cos(E(x+δ),E(x)) direct ↓', 'val cos after narrowband ↓', 'val SNR of δ (dB)']): a.set_title(t); a.set_xlabel('step')
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig('plots/learning_curves.png', dpi=100); plt.show()
pd.DataFrame({k: d.dropna(subset=['val_cos']).iloc[-1][['step','val_cos','val_cos_nb','val_perc','val_snr']] for k, d in runs.items()}).T""")
md("## 6. Evaluation under the threat model (Table 3.1, scaled down)\n`scripts/run_eval.py`: 40 LibriSpeech test speakers × 2 references (4–10 s) each, enrolment = 3 genuine utterances from other chapters. **No cloner is run** (see §11): the outcome is whether a verifier accepts the *protected reference itself* as the victim — the identity signal a cloner would copy. Thresholds τ_v at the clean EER point. Seen verifier: ECAPA. **Held-out (black-box transfer): WavLM-Base+ SV and Resemblyzer (GE2E).**")
code("""from vpm.report import thresholds, acceptance, ar_table, rho, h5
sc = pd.concat([pd.read_csv(f) for f in ['results/eval_scores.csv', 'results/eval_scores_e2.csv'] if os.path.exists(f)], ignore_index=True); q = pd.read_csv('results/eval_quality.csv')
thr = thresholds(sc); display(thr)
M = acceptance(sc, thr); T = ar_table(M); T.to_csv('results/ar_table.csv', index=False)
print('quality of protected references (direct, before any channel):')
Q = q.groupby('protection')[['snr_db','pesq_wb','stoi','linf']].agg(['mean','median']).round(3); Q.to_csv('results/quality_table.csv'); Q""")
code("""piv = T.pivot_table(index=['condition','channel'], columns=['verifier','protection'], values='AR_prot').round(2)
piv.to_csv('results/AR_pivot.csv'); piv""")
code("""main = T[T.protection.isin(['B1_gaussian','B4_teacher_PGD-EOT','B5_GM_generic','B6_GM_KE','B6_GM_KE_INT8(ORT)','GS_KE(0ms)'])]
show = main.assign(dAR_CI=main.apply(lambda r: f"{r.dAR:+.2f} [{r.dAR_lo:+.2f},{r.dAR_hi:+.2f}]", axis=1))
show.pivot_table(index=['condition','channel'], columns=['verifier','protection'], values='dAR_CI', aggfunc='first')""")
code("""fig, ax = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
for a, (v, g) in zip(ax, T.groupby('verifier')):
    p = g.pivot_table(index='channel', columns='protection', values='AR_prot'); p.plot.bar(ax=a, legend=a is ax[0]); a.set_title(f'{v}: acceptance of protected refs'); a.set_ylim(0, 1.05)
plt.tight_layout(); plt.savefig('plots/acceptance_by_condition.png', dpi=100); plt.show()""")
code("""# similarity shift distribution (held-out WavLM)
fig, ax = plt.subplots(figsize=(8, 4))
for p, g in M[(M.channel=='none') & M.verifier.str.startswith('WavLM')].groupby('protection'): ax.hist(g.score, bins=25, alpha=.45, label=p)
ax.axvline(thr.loc['WavLM-SV(held-out)','tau'], color='k', ls='--', label='τ (EER)'); ax.set_title('WavLM-SV score vs enrolment (direct)'); ax.legend(fontsize=7)
plt.savefig('plots/wavlm_score_hist.png', dpi=100); plt.show()""")
md("### Protection retention ρ (Eq. 15) — deployed (FP32 / INT8 streaming) vs PGD-EOT teacher, on the 40 references where the teacher was run")
code("""R = pd.concat([rho(M, d).assign(deployed=d) for d in ['B6_GM_KE','B6_GM_KE_INT8(ORT)']]); R.to_csv('results/rho.csv', index=False)
R.pivot_table(index=['verifier','channel'], columns='deployed', values='rho').round(2)""")
md("### H5 — utility non-inferiority (ASR ΔWER/ΔCER, Whisper base.en) and Swahili (FLEURS sw_ke) utility")
code("""A = pd.read_csv('results/asr_wer.csv'); H5 = h5(A); H5.to_csv('results/h5_asr.csv', index=False); display(H5.round(2))
L = pd.read_csv('results/language_utility.csv'); LU = L.groupby('lang')[['snr_db','pesq_wb','stoi','ecapa_cos_clean_vs_prot','wavlm_cos_clean_vs_prot']].mean().round(3); LU.to_csv('results/language_table.csv'); LU""")
md("## 7. ONNX export and optimisation (§3.7.1)\nTwo graphs: the **streaming step** (one 10 ms block + fixed-size state in/out, what a PipeWire filter or Raspberry Pi service would call) and the **offline** full-utterance graph. Equivalence vs PyTorch on 8 validation probes (target ≤1e-3); then ORT graph optimisation, dynamic INT8 (naive = all ops; sensitivity-aware = DFT/mel front-end and output layer kept FP32) and static QDQ INT8 (calibrated on real streaming states from 32 training utterances).")
code("""if RERUN: os.system('PYTHONPATH=src python scripts/export_all.py models/gm_ke_lam0.1/best.pt models/onnx/gm_ke --bench')
eq = pd.read_csv('models/onnx/gm_ke/equivalence.csv'); display(eq.drop(columns='path'))
bm = pd.read_csv('models/onnx/gm_ke/latency_benchmark.csv'); display(bm.round(4))""")
code("""import onnxruntime as ort
from vpm.export import ort_session, ort_stream, load_generator
g = load_generator('models/gm_ke_lam0.1/best.pt'); sgm = StreamingGenerator(g)
xv = sf.read(m[m.split=='val'].clean_path.iloc[0], dtype='float32')[0][:16000*2]
ref = stream_process(sgm, torch.from_numpy(xv)).numpy()
for f in ['step_fp32.onnx', 'step_int8_dynamic.onnx', 'step_int8_static_qdq.onnx']:
    o = ort_stream(ort_session(f'models/onnx/gm_ke/{f}'), xv, g.L, g.cfg['C'], sgm.S)
    print(f'{f:28s} max|Δ|={np.abs(o-ref).max():.2e}  rel={np.linalg.norm(o-ref)/np.linalg.norm(ref):.4f}  ||δ||∞={np.abs(o).max():.4f} ≤ ε={g.eps}')
fig, ax = plt.subplots(1, 2, figsize=(14, 3.5))
bm.plot.bar(x='model', y=['p50_ms','p99_ms'], ax=ax[0], title='per-10 ms-block compute latency (1 thread, x86)'); ax[0].axhline(10, color='r', ls='--')
bm.plot.bar(x='model', y='size_kb', ax=ax[1], title='model size (KB)', legend=False); plt.tight_layout(); plt.savefig('plots/onnx_benchmark.png', dpi=100); plt.show()""")
md("## 8. Listening example\nClean vs protected (G-M, FP32) — audio players below.")
code("""from IPython.display import Audio, display
with torch.no_grad(): d = g(torch.from_numpy(xv[:len(xv)//160*160])[None])[0].numpy()
sf.write('results/example_clean.wav', xv, 16000); sf.write('results/example_protected_gm.wav', np.clip(xv[:len(d)]+d, -1, 1), 16000)
display(Audio('results/example_clean.wav')); display(Audio('results/example_protected_gm.wav'))
fig = plot_spectrograms(['results/example_clean.wav', 'results/example_protected_gm.wav'], ['clean', 'protected (G-M, ε=0.008)'], 'plots/example_spectrograms.png'); plt.show()""")
md(open("/tmp/summary.md").read() + "\n\n" + open("/tmp/caveats.md").read())
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/Voice_Protection_Model.ipynb")
