"""E2 (white-box adaptive): removal network trained on generator outputs, applied before verification."""
import json, sys, numpy as np, pandas as pd, soundfile as sf, torch, torch.nn.functional as F
from vpm.encoders import ECAPA, WavLMSV, Resemblyzer
from vpm.export import load_generator, ort_session, ort_stream
from vpm.generator import StreamingGenerator
from vpm.removal import load_removal
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
P = json.load(open("results/eval_protocol.json")); m = pd.read_csv("data/manifests/all.csv").set_index("utt_id")
g = load_generator("models/gm_ke_lam0.1/best.pt"); sg = StreamingGenerator(g); sess = ort_session("models/onnx/gm_ke/step_int8_dynamic.onnx")
rem = load_removal("models/removal_net_E2.pt"); ec = ECAPA()
V = {"ECAPA(seen)": lambda ws: torch.cat([ec(w[None]) for w in ws]).detach(), "Resemblyzer(held-out)": Resemblyzer(), "WavLM-SV(held-out)": WavLMSV()}
load = lambda u: torch.tensor(sf.read(m.loc[u, "clean_path"], dtype="float32")[0]); pad = lambda x: F.pad(x, (0, (-len(x)) % 160))
enr = {v: {s: F.normalize(F.normalize(f([torch.tensor(sf.read(p, dtype="float32")[0][:16000 * 8]) for p in ps]), dim=-1).mean(0), dim=0) for s, ps in P["enroll"].items()} for v, f in V.items()}
prot = {"B0_unprotected": lambda x: x, "B6_GM_KE": lambda x: (x + g(pad(x)[None])[0, :len(x)]).clamp(-1, 1).detach(),
        "B6_GM_KE_INT8(ORT)": lambda x: (x + torch.from_numpy(ort_stream(sess, pad(x).numpy(), g.L, g.cfg["C"], sg.S))[:len(x)]).clamp(-1, 1)}
rows = []; spks = sorted(P["enroll"])
for u in P["refs"]:
    x = load(u); s0 = m.loc[u, "speaker"]
    for pn, pf in prot.items():
        with torch.no_grad(): y = rem(pf(x))
        for v, f in V.items():
            e = F.normalize(f([y])[0], dim=0); sc = {s: float(e @ enr[v][s]) for s in spks}
            rows.append(dict(ref=u, speaker=s0, protection=pn, channel="E2_removal_net(white-box)", condition="E2 adaptive white-box", verifier=v,
                             score=sc[s0], max_impostor=max(sc[s] for s in spks if s != s0), impostor_scores=json.dumps([round(sc[s], 4) for s in spks if s != s0])))
pd.DataFrame(rows).to_csv("results/eval_scores_e2.csv", index=False); print("done", len(rows))
