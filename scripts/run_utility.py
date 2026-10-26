"""Utility: H5 ΔWER (English, Whisper base.en, LibriSpeech test refs) and Swahili (FLEURS sw_ke) quality +
embedding displacement. usage: PYTHONPATH=src python scripts/run_utility.py MAIN.pt INT8.onnx"""
import sys, json, re, numpy as np, pandas as pd, soundfile as sf, torch, torch.nn.functional as F, jiwer
from transformers import pipeline
from vpm.export import load_generator, ort_session, ort_stream
from vpm.generator import StreamingGenerator
from vpm.evaluate import quality
from vpm.encoders import ECAPA, WavLMSV
from vpm import channels as ch
torch.set_num_threads(2)
g = load_generator(sys.argv[1]); sess = ort_session(sys.argv[2]); sg = StreamingGenerator(g)
pad = lambda x: F.pad(x, (0, (-len(x)) % 160))
P = {"B0_unprotected": lambda x: x,
     "B6_GM_KE": lambda x: (x + g(pad(x)[None])[0, :len(x)]).clamp(-1, 1).detach(),
     "B6_GM_KE_INT8(ORT)": lambda x: (x + torch.from_numpy(ort_stream(sess, pad(x).numpy(), g.L, g.cfg["C"], sg.S))[:len(x)]).clamp(-1, 1)}
m = pd.read_csv("data/manifests/all.csv")
te = m[(m.split == "test") & (m.dur >= 4) & (m.dur <= 10)]
proto = {"refs": [u for spk, g in te.groupby("speaker") for u in g.sample(2, random_state=0).utt_id]}  # same refs as run_eval.py
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")
norm = lambda s: re.sub(r"[^a-z' ]", " ", s.lower()).split()
rows = []
ref = m[m.utt_id.isin(proto["refs"])]
for _, r in ref.iterrows():
    x = torch.tensor(sf.read(r.clean_path, dtype="float32")[0])
    for pn, pf in P.items():
        xp = pf(x)
        for cn, cf in {"direct": lambda y: y, "g711_mulaw_8k": lambda y: ch.narrowband(y[None])[0]}.items():
            y = cf(xp).detach().numpy()
            hyp = asr({"raw": y, "sampling_rate": 16000})["text"]
            h, t = " ".join(norm(hyp)), " ".join(norm(r.text))
            rows.append(dict(utt=r.utt_id, speaker=r.speaker, protection=pn, channel=cn, wer=jiwer.wer(t, h), cer=jiwer.cer(t, h), hyp=hyp))
    print(len(rows), flush=True)
pd.DataFrame(rows).to_csv("results/asr_wer.csv", index=False)
# ---- Swahili (Kenyan, FLEURS sw_ke): no speaker IDs → utility + embedding displacement only (no cloning)
ec, wl = ECAPA(), WavLMSV(); sw = m[m.split == "sw_test"].sample(60, random_state=0); en = ref.sample(60, random_state=0, replace=len(ref) < 60)
srows = []
for lang, df in [("sw (FLEURS sw_ke)", sw), ("en (LibriSpeech test)", en)]:
    for _, r in df.iterrows():
        x = torch.tensor(sf.read(r.clean_path, dtype="float32")[0])[: 16000 * 10]
        xp = P["B6_GM_KE"](x)
        with torch.no_grad():
            c1 = F.cosine_similarity(ec(x[None]), ec(xp[None])).item()
        c2 = float(wl([x]) @ wl([xp]).T)
        srows.append(dict(lang=lang, utt=r.utt_id, ecapa_cos_clean_vs_prot=c1, wavlm_cos_clean_vs_prot=c2, **quality(x.numpy(), xp.numpy())))
pd.DataFrame(srows).to_csv("results/language_utility.csv", index=False); print("done")
