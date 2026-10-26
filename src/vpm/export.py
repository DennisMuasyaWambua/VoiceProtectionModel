"""ONNX export (offline + streaming step), PyTorch↔ORT equivalence, INT8 quantisation, benchmarks (§3.7)."""
from __future__ import annotations
import os, time, numpy as np, torch, onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
from .generator import Generator, StreamingGenerator, H

def load_generator(path):
    ck = torch.load(path, map_location="cpu"); g = Generator(**ck["cfg"]); g.load_state_dict(ck["state"]); return g.eval()

STEP_INPUTS = ["block", "prev", "mode", "conv_state", "gru_state"]
STEP_OUTPUTS = ["delta", "prev_out", "conv_state_out", "gru_state_out"]

def export_step(g: Generator, path):
    sg = StreamingGenerator(g).eval(); prev, cs, gs = sg.init_state()
    args = (torch.zeros(1, H), prev, torch.tensor([[0., 0., 1.]]), cs, gs)
    torch.onnx.export(sg, args, path, input_names=STEP_INPUTS, output_names=STEP_OUTPUTS, opset_version=17, dynamo=False)
    return sg

def export_offline(g: Generator, path):
    x = torch.zeros(1, H * 100); mode = torch.tensor([[0., 0., 1.]])
    torch.onnx.export(g, (x, mode), path, input_names=["audio", "mode"], output_names=["delta"], opset_version=17,
                      dynamic_axes={"audio": {1: "n"}, "delta": {1: "n"}}, dynamo=False)

def ort_session(path, threads=1):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

def ort_stream(sess, x: np.ndarray, L: int, C: int, S: int, mode=(0., 0., 1.), times=None):
    """Run the streaming ONNX model block by block. Returns δ aligned to x."""
    x = np.pad(x.astype(np.float32), (0, L * H)).reshape(-1, H)
    prev = np.zeros((1, H), np.float32); cs = np.zeros((1, C, S), np.float32); gs = np.zeros((1, 1, C), np.float32)
    m = np.array([mode], np.float32); outs = []
    for b in x:
        t = time.perf_counter()
        d, prev, cs, gs = sess.run(None, {"block": b[None], "prev": prev, "mode": m, "conv_state": cs, "gru_state": gs})
        if times is not None: times.append(time.perf_counter() - t)
        outs.append(d[0])
    return np.concatenate(outs[L:])

class StepCalib(CalibrationDataReader):
    """Feeds real streaming states from calibration utterances (static QDQ quantisation)."""
    def __init__(self, sg: StreamingGenerator, wavs):
        feeds = []
        for w in wavs:
            prev, cs, gs = sg.init_state(); xin = torch.from_numpy(w[: len(w) // H * H]).view(-1, H)
            with torch.no_grad():
                for i, b in enumerate(xin):
                    f = dict(block=b[None].numpy(), prev=prev.numpy(), mode=np.array([[0, 0, 1]], np.float32), conv_state=cs.numpy(), gru_state=gs.numpy())
                    if i % 4 == 0: feeds.append(f)
                    _, prev, cs, gs = sg(b[None], prev, torch.tensor([[0., 0., 1.]]), cs, gs)
        self.it = iter(feeds)
    def get_next(self): return next(self.it, None)

def quantize_all(fp32, out_dir, sg, calib_wavs):
    pre = os.path.join(out_dir, "step_fp32_pre.onnx"); quant_pre_process(fp32, pre)
    # Sensitivity-aware: keep the DFT/mel front-end and the output projection in FP32
    # (naive full INT8 gave 26–56 % relative error on δ; with exclusions ≈1 %).
    names = [n.name for n in onnx.load(pre).graph.node if n.op_type in ("MatMul", "Gemm", "Conv", "GRU")]
    excl = [n for n in names if "feat" in n or "/out/" in n]
    dyn = os.path.join(out_dir, "step_int8_dynamic.onnx")
    quantize_dynamic(pre, dyn, weight_type=QuantType.QInt8, per_channel=True, nodes_to_exclude=excl)
    naive = os.path.join(out_dir, "step_int8_dynamic_naive.onnx")
    quantize_dynamic(pre, naive, weight_type=QuantType.QInt8)
    sta = os.path.join(out_dir, "step_int8_static_qdq.onnx")
    quantize_static(pre, sta, StepCalib(sg, calib_wavs), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"], nodes_to_exclude=excl)
    opt = os.path.join(out_dir, "step_fp32_ortopt.onnx")
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.optimized_model_filepath = opt
    ort.InferenceSession(fp32, so, providers=["CPUExecutionProvider"])
    return dict(dynamic=dyn, static=sta, ortopt=opt, naive=naive)

def bench(path, L, C, S, n_blocks=1000, threads=1):
    sess = ort_session(path, threads); times = []
    x = (np.random.default_rng(0).standard_normal(n_blocks * H) * 0.05).astype(np.float32)
    ort_stream(sess, x[: 50 * H], L, C, S)  # warm-up
    ort_stream(sess, x, L, C, S, times=times); t = np.array(times) * 1e3
    return dict(model=os.path.basename(path), size_kb=os.path.getsize(path) / 1024, p50_ms=float(np.median(t)),
                p99_ms=float(np.percentile(t, 99)), mean_ms=float(t.mean()), rtf=float(t.mean() / 10.0))
