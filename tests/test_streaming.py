import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vpm.generator import G_S, G_M, StreamingGenerator, stream_process

def test_stream_equals_offline():
    torch.manual_seed(0)
    for g in (G_S(), G_M()):
        g.eval(); x = torch.randn(1, 160 * 50) * 0.05
        with torch.no_grad(): off = g(x)[0]
        st = stream_process(StreamingGenerator(g).eval(), x[0])
        err = (off - st).abs().max().item()
        print(type(g).__name__, g.cfg, "params", g.n_params(), "max|offline-stream|", err)
        assert err < 1e-5
        assert off.abs().max() <= g.eps + 1e-7

if __name__ == "__main__": test_stream_equals_offline()
