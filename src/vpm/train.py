"""Train the streaming generator with the channel-aware EOT objective (Eq. 11), curriculum (§3.6.4).

usage: python -m vpm.train --variant GM --tdist ke --steps 1200 --out models/gm_ke
"""
from __future__ import annotations
import argparse, json, math, os, random, time
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn.functional as F, torchaudio.functional as AF, yaml
from .generator import G_S, G_M, MODES
from .encoders import ECAPA
from .losses import perceptual_loss, snr_db
from . import channels as ch

def load_crop(path, n, rng):
    info = sf.info(path); L = info.frames
    s = rng.randint(0, max(0, L - n))
    x, _ = sf.read(path, start=s, frames=n, dtype="float32")
    if len(x) < n: x = np.pad(x, (0, n - len(x)))
    return x

def generic_t(x, rng):
    """T_gen baseline: MP3 64–128k, resampling 8–22 kHz, low-pass 4–8 kHz, 8–16-bit quantisation (STE)."""
    op = rng.choice(["mp3", "resample", "lowpass", "quant"])
    if op == "mp3": y = ch.codec_ste(x, "libmp3lame", rng.choice(["64k", "96k", "128k"]))
    elif op == "resample":
        sr = rng.choice([8000, 11025, 22050]); y = AF.resample(AF.resample(x, 16000, sr), sr, 16000)[..., :x.shape[-1]]
    elif op == "lowpass": y = ch._fft_bandpass(x, 0, rng.uniform(4000, 8000))
    else:
        b = rng.randint(8, 16); q = 2 ** (b - 1); y = ch._ste(x, torch.round(x * q) / q)
    return y, op

def transform(x, xc, rng, tdist, mode, stage):
    if tdist == "gen": return generic_t(x, rng)
    if stage == 0: fams = [("clean", 1.0)]
    elif stage == 1: fams = [("narrowband", .3), ("wideband", .2), ("voicenote", .35), ("clean", .15)]
    else:
        if mode == "mobile": fams = [("narrowband", .3), ("wideband", .2), ("voicenote", .35), ("clean", .15)]
        elif mode == "pa_live": fams = [("pa", 1.0)]
        else: fams = ch.FAMILIES
    return ch.sample_ke(x, xc, rng, families=fams)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="GM"); ap.add_argument("--tdist", default="ke")
    ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml")); T = cfg["train"]
    lam = T["lambda_perc"] if a.lam is None else a.lam; eps = cfg["generator"]["eps"] if a.eps is None else a.eps
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    m = pd.read_csv(f"{cfg['paths']['manifests']}/all.csv")
    tr = m[m.split == "train"].clean_path.tolist(); va = m[m.split == "val"].clean_path.tolist()
    N = int(T["crop_s"] * 16000) // 160 * 160
    vr = random.Random(123); vb = torch.tensor(np.stack([load_crop(p, N, vr) for p in vr.sample(va, 16)]))
    G = (G_M if a.variant == "GM" else G_S)(eps)
    enc = ECAPA()
    opt = torch.optim.AdamW(G.parameters(), lr=T["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    with torch.no_grad(): ve_clean = enc(vb)
    log, best = [], 1e9
    json.dump(dict(vars(a), lam=lam, eps=eps, gen_cfg=G.cfg, params=G.n_params(), macs_per_s=G.macs_per_second()), open(f"{a.out}/train_cfg.json", "w"), indent=1)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        stage = 0 if step < 0.15 * a.steps else (1 if step < 0.4 * a.steps else 2)
        x = torch.tensor(np.stack([load_crop(p, N, rng) for p in rng.sample(tr, T["batch"])]))
        modes = [rng.choice(MODES) for _ in range(len(x))]
        mo = torch.stack([F.one_hot(torch.tensor(MODES.index(md)), 3).float() for md in modes])
        G.train(); d = G(x, mo); xp = (x + d).clamp(-1, 1)
        with torch.no_grad(): e_clean = enc(x)
        ys = [transform(xp[i:i + 1], x[i:i + 1], rng, a.tdist, modes[i], stage)[0] for i in range(len(x))]
        e_adv = enc(torch.cat(ys))
        l_adv = F.cosine_similarity(e_adv, e_clean).mean()
        l_perc = perceptual_loss(x, xp, T["gamma_mask"])
        loss = l_adv + lam * l_perc
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(G.parameters(), T["grad_clip"]); opt.step(); sched.step()
        rec = dict(step=step, stage=stage, loss=loss.item(), cos_train=l_adv.item(), perc=l_perc.item(),
                   snr=snr_db(x, d.detach()).mean().item(), sec=time.time() - t0)
        if step % 50 == 0 or step == a.steps:
            G.eval()
            with torch.no_grad():
                vd = G(vb); vxp = (vb + vd).clamp(-1, 1)
                vc = F.cosine_similarity(enc(vxp), ve_clean).mean().item()
                vnb = F.cosine_similarity(enc(ch.narrowband(vxp)), enc(ch.narrowband(vb))).mean().item()
                vp = perceptual_loss(vb, vxp, T["gamma_mask"]).item()
            rec.update(val_cos=vc, val_cos_nb=vnb, val_perc=vp, val_obj=vc + lam * vp, val_snr=snr_db(vb, vd).mean().item())
            if rec["val_obj"] < best and stage == 2:
                best = rec["val_obj"]; torch.save(dict(state=G.state_dict(), cfg=G.cfg, step=step), f"{a.out}/best.pt")
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        log.append(rec)
        if step % 50 == 0: pd.DataFrame(log).to_csv(f"{a.out}/log.csv", index=False)
    torch.save(dict(state=G.state_dict(), cfg=G.cfg, step=a.steps), f"{a.out}/last.pt")
    if not os.path.exists(f"{a.out}/best.pt"): torch.save(dict(state=G.state_dict(), cfg=G.cfg, step=a.steps), f"{a.out}/best.pt")
    pd.DataFrame(log).to_csv(f"{a.out}/log.csv", index=False)

if __name__ == "__main__": main()
