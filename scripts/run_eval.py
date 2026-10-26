"""Full evaluation (scaled-down E1/E2/E3/E4/E6-sim) → results/*.csv
usage: PYTHONPATH=src python scripts/run_eval.py --main models/gm_ke_lam0.1/best.pt ..."""
import argparse, json, os, random, time, sys
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn.functional as F
from vpm.encoders import ECAPA, WavLMSV, Resemblyzer
from vpm.export import load_generator, ort_session, ort_stream
from vpm.generator import StreamingGenerator
from vpm.teacher import pgd_eot
from vpm.evaluate import eer_threshold, quality, cluster_bootstrap
from vpm import channels as ch

ap = argparse.ArgumentParser()
ap.add_argument("--main", required=True); ap.add_argument("--gen_baseline"); ap.add_argument("--gs")
ap.add_argument("--int8"); ap.add_argument("--removal"); ap.add_argument("--refs_per_spk", type=int, default=2)
ap.add_argument("--teacher_K", type=int, default=60); ap.add_argument("--teacher_lam", type=float, default=0.1)
ap.add_argument("--out", default="results"); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--no_wavlm", action="store_true")
a = ap.parse_args(); torch.set_num_threads(a.threads)
m = pd.read_csv("data/manifests/all.csv")
te = m[(m.split == "test") & (m.dur >= 4) & (m.dur <= 10)].copy()
rng = random.Random(0); refs, enroll = [], {}
for spk, g in m[m.split == "test"].groupby("speaker"):
    cand = te[te.speaker == spk]; r = cand.sample(a.refs_per_spk, random_state=0)
    refs += r.to_dict("records")
    rest = g.drop(r.index, errors="ignore"); rest = rest[~rest.utt_id.isin(r.utt_id)]
    # enrol (genuine x') from other chapters where possible
    other = rest[~rest.utt_id.str.split("-").str[1].isin(r.utt_id.str.split("-").str[1])]
    enroll[spk] = (other if len(other) >= 3 else rest).sample(3, random_state=1).clean_path.tolist()
print("refs", len(refs), "speakers", len(enroll), flush=True)
load = lambda p: torch.tensor(sf.read(p, dtype="float32")[0])
def pad160(x): return F.pad(x, (0, (-len(x)) % 160))

ecapa = ECAPA()
verifiers = {"ECAPA(seen)": lambda ws: torch.cat([ecapa(w[None]) for w in ws]).detach(), "Resemblyzer(held-out)": Resemblyzer()}
if not a.no_wavlm: verifiers["WavLM-SV(held-out)"] = WavLMSV()

# ---- enrolment embeddings & clean thresholds
enr = {v: {} for v in verifiers}
for spk, ps in enroll.items():
    ws = [load(p)[:16000 * 8] for p in ps]
    for v, f in verifiers.items(): enr[v][spk] = F.normalize(F.normalize(f(ws), dim=-1).mean(0), dim=0)
spks = sorted(enroll)

# ---- protections
G = {"B6_GM_KE": load_generator(a.main)}
if a.gen_baseline: G["B5_GM_generic"] = load_generator(a.gen_baseline)
if a.gs: G["GS_KE(0ms)"] = load_generator(a.gs)
eps = G["B6_GM_KE"].eps
prot = {"B0_unprotected": lambda x, i: x}
def gauss(x, i):
    g = torch.Generator().manual_seed(i); return x + (torch.randn(x.shape, generator=g) * eps / 2).clamp(-eps, eps)
prot["B1_gaussian"] = gauss
for k, g in G.items(): prot[k] = (lambda g: lambda x, i: (x + g(pad160(x)[None])[0, : len(x)].detach()).clamp(-1, 1))(g)
if a.int8:
    sess = ort_session(a.int8); gm = G["B6_GM_KE"]; sg = StreamingGenerator(gm)
    prot["B6_GM_KE_INT8(ORT)"] = lambda x, i: (x + torch.from_numpy(ort_stream(sess, pad160(x).numpy(), gm.L, gm.cfg["C"], sg.S))[: len(x)]).clamp(-1, 1)
teacher_cache = {}
def teacher(x, i):
    if i not in teacher_cache:
        xx = pad160(x)[None]; d = pgd_eot(xx, ecapa, eps=eps, K=a.teacher_K, M=2, lam=a.teacher_lam, seed=i)
        teacher_cache[i] = (xx + d)[0, : len(x)].clamp(-1, 1)
    return teacher_cache[i]
prot["B4_teacher_PGD-EOT"] = teacher
if a.removal:
    from vpm.removal import load_removal; rem = load_removal(a.removal)

# ---- channels (E-conditions, scaled-down)
def seen_ke(x, i): return ch.sample_ke(x[None], x[None], random.Random(1000 + i))[0][0]
def e6_sim(x, i):  # unseen chain: re-recording (new RIR draws) → Opus 12k → G.711
    y = ch.pa_rerecord(x[None], x[None], random.Random(5000 + i)); return ch.narrowband(ch.codec_ste(y, "libopus", "12k"))[0]
H = ch.heldout_channels(); R = ch.removal_attacks()
channels = {"none": ("E1/E3 direct", lambda x, i: x), "seen_T_KE_sample": ("E1 seen channel", seen_ke)}
for k in ["g711_mulaw_8k", "opus_6k", "opus_48k", "chain_opus12k_g711"]: channels[k] = ("E3 held-out codec", (lambda f: lambda x, i: f(x[None])[0])(H[k]))
for k, f in R.items(): channels[k] = ("E4 generic removal", (lambda f: lambda x, i: f(x[None])[0])(f))
channels["E6sim_rerecord_opus_g711"] = ("E6-sim chained", e6_sim)
if a.removal: channels["E2_removal_net(white-box)"] = ("E2 adaptive white-box", lambda x, i: rem(x))

rows, qrows = [], []; t0 = time.time()
for i, r in enumerate(refs):
    x = load(r["clean_path"])
    for pn, pf in prot.items():
        if pn.startswith("B4") and i % a.refs_per_spk != 0: continue  # teacher on 1 ref/speaker (compute)
        xp = pf(x, i).detach()
        qrows.append(dict(ref=r["utt_id"], speaker=r["speaker"], protection=pn, **quality(x.numpy(), xp.numpy())))
        sf.write(f"/tmp/vp_{pn[:2]}.wav", xp.numpy(), 16000)
        for cn, (cond, cf) in channels.items():
            with torch.no_grad(): y = cf(xp, i).detach()
            for v, f in verifiers.items():
                e = F.normalize(f([y])[0], dim=0)
                sc = {s: float(e @ enr[v][s]) for s in spks}
                rows.append(dict(ref=r["utt_id"], speaker=r["speaker"], protection=pn, channel=cn, condition=cond, verifier=v,
                                 score=sc[r["speaker"]], max_impostor=max(sc[s] for s in spks if s != r["speaker"]),
                                 impostor_scores=json.dumps([round(sc[s], 4) for s in spks if s != r["speaker"]])))
    print(f"ref {i+1}/{len(refs)}  {time.time()-t0:.0f}s", flush=True)
    if (i + 1) % 10 == 0: pd.DataFrame(rows).to_csv(f"{a.out}/eval_scores.csv", index=False); pd.DataFrame(qrows).to_csv(f"{a.out}/eval_quality.csv", index=False)
pd.DataFrame(rows).to_csv(f"{a.out}/eval_scores.csv", index=False); pd.DataFrame(qrows).to_csv(f"{a.out}/eval_quality.csv", index=False)
json.dump(dict(refs=[r["utt_id"] for r in refs], enroll=enroll, eps=eps), open(f"{a.out}/eval_protocol.json", "w"), indent=1)
print("done", time.time() - t0)
