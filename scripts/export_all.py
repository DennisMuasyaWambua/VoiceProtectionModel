"""Export the trained generator(s) to ONNX, verify equivalence, quantise, benchmark.
usage: PYTHONPATH=src python scripts/export_all.py models/gm_ke_lam0.1/best.pt models/onnx/gm [--bench]"""
import sys, os, json, numpy as np, pandas as pd, soundfile as sf, torch
from vpm.export import *
from vpm.generator import StreamingGenerator, stream_process
torch.set_num_threads(1)
ck, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
g = load_generator(ck); sg = export_step(g, f"{out}/step_fp32.onnx"); export_offline(g, f"{out}/offline_fp32.onnx")
m = pd.read_csv("data/manifests/all.csv")
probe = [sf.read(p, dtype="float32")[0][:16000 * 3] for p in m[m.split == "val"].clean_path.sample(8, random_state=0)]
calib = [sf.read(p, dtype="float32")[0][:16000 * 3] for p in m[m.split == "train"].clean_path.sample(32, random_state=0)]
q = quantize_all(f"{out}/step_fp32.onnx", out, sg, calib)
C, S, L = g.cfg["C"], sg.S, g.L; rows = []
for name, path in [("fp32", f"{out}/step_fp32.onnx"), ("fp32_ort_optimized", q["ortopt"]), ("int8_dynamic_naive(all ops)", q["naive"]), ("int8_dynamic", q["dynamic"]), ("int8_static_qdq", q["static"])]:
    s = ort_session(path); errs, rels = [], []
    for x in probe:
        x = x[: len(x) // 160 * 160]; ref = stream_process(sg, torch.from_numpy(x)).numpy(); o = ort_stream(s, x, L, C, S)
        errs.append(np.abs(o - ref).max()); rels.append(np.linalg.norm(o - ref) / np.linalg.norm(ref))
    off = None
    rows.append(dict(model=name, path=path, size_kb=os.path.getsize(path) / 1024, max_abs_err_vs_torch=float(max(errs)), rel_l2_err=float(np.mean(rels)), within_1e3_tol=bool(max(errs) < 1e-3)))
x = probe[0][: len(probe[0]) // 160 * 160]
so = ort_session(f"{out}/offline_fp32.onnx").run(None, {"audio": x[None], "mode": np.array([[0, 0, 1]], np.float32)})[0][0]
with torch.no_grad(): ref = g(torch.from_numpy(x)[None])[0].numpy()
rows.append(dict(model="offline_fp32 (full-utterance graph)", path=f"{out}/offline_fp32.onnx", size_kb=os.path.getsize(f"{out}/offline_fp32.onnx") / 1024, max_abs_err_vs_torch=float(np.abs(so - ref).max()), rel_l2_err=float(np.linalg.norm(so - ref) / np.linalg.norm(ref)), within_1e3_tol=bool(np.abs(so - ref).max() < 1e-3)))
pd.DataFrame(rows).to_csv(f"{out}/equivalence.csv", index=False); print(pd.DataFrame(rows).drop(columns="path").to_string())
torch.save(dict(state=g.state_dict(), cfg=g.cfg), f"{out}/generator_fp32.pt")
if "--bench" in sys.argv:
    b = []
    for name, path in [("pytorch_eager", None), ("fp32", f"{out}/step_fp32.onnx"), ("fp32_ort_optimized", q["ortopt"]), ("int8_dynamic", q["dynamic"]), ("int8_static_qdq", q["static"])]:
        if path is None:
            import time; prev, cs, gs = sg.init_state(); t = []; md = torch.tensor([[0., 0., 1.]])
            with torch.no_grad():
                for i in range(1050):
                    t0 = time.perf_counter(); _, prev, cs, gs = sg(torch.randn(1, 160) * .05, prev, md, cs, gs); t.append(time.perf_counter() - t0)
            t = np.array(t[50:]) * 1e3; b.append(dict(model=name, size_kb=os.path.getsize(ck) / 1024, p50_ms=np.median(t), p99_ms=np.percentile(t, 99), mean_ms=t.mean(), rtf=t.mean() / 10))
        else: b.append(dict(bench(path, L, C, S, 3000), model=name))
    bd = pd.DataFrame(b); bd["params"] = g.n_params(); bd["algorithmic_latency_ms"] = 10 + 10 * L
    bd["e2e_ms_excl_buffers(alg+p99)"] = bd.algorithmic_latency_ms + bd.p99_ms
    bd.to_csv(f"{out}/latency_benchmark.csv", index=False); print(bd.to_string())
