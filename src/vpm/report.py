"""Analysis of evaluation outputs: thresholds, acceptance rates, ΔAR with speaker-cluster bootstrap, ρ, H5."""
import json, numpy as np, pandas as pd
from .evaluate import eer_threshold, cluster_bootstrap

def thresholds(sc):
    out = {}
    for v, g in sc[(sc.protection == "B0_unprotected") & (sc.channel == "none")].groupby("verifier"):
        imp = np.concatenate([json.loads(s) for s in g.impostor_scores])
        t, eer = eer_threshold(g.score.values, imp); out[v] = dict(tau=t, clean_EER=eer, n_genuine=len(g), n_impostor=len(imp))
    return pd.DataFrame(out).T

def acceptance(sc, thr):
    sc = sc.copy(); sc["accept"] = sc.apply(lambda r: float(r.score >= thr.loc[r.verifier, "tau"]), axis=1)
    base = sc[sc.protection == "B0_unprotected"][["ref", "channel", "verifier", "accept", "score"]].rename(columns={"accept": "acc0", "score": "s0"})
    m = sc.merge(base, on=["ref", "channel", "verifier"]); m["dAR"] = m.acc0 - m.accept; m["dSIM"] = m.s0 - m.score
    return m

def ar_table(m, n_boot=1000):
    rows = []
    for (c, ch, p, v), g in m.groupby(["condition", "channel", "protection", "verifier"]):
        d, lo, hi = cluster_bootstrap(g, "dAR", n=n_boot) if p != "B0_unprotected" else (0, 0, 0)
        rows.append(dict(condition=c, channel=ch, protection=p, verifier=v, n=len(g), AR_unprot=g.acc0.mean(), AR_prot=g.accept.mean(),
                         dAR=d, dAR_lo=lo, dAR_hi=hi, SIM_prot=g.score.mean(), dSIM=g.dSIM.mean()))
    return pd.DataFrame(rows)

def rho(m, deployed, teacher="B4_teacher_PGD-EOT"):
    """ρ = ΔAR_deployed / ΔAR_teacher on the refs where the teacher was run (Eq. 15)."""
    refs = m[m.protection == teacher].ref.unique(); s = m[m.ref.isin(refs)]
    out = []
    for (ch, v), g in s.groupby(["channel", "verifier"]):
        dt = g[g.protection == teacher].dAR.mean(); dd = g[g.protection == deployed].dAR.mean()
        out.append(dict(channel=ch, verifier=v, dAR_teacher=dt, dAR_deployed=dd, rho=dd / dt if dt > 0 else np.nan))
    return pd.DataFrame(out)

def h5(asr, prot="B6_GM_KE", n_boot=2000):
    """H5 non-inferiority: one-sided upper 95 % bound of ΔWER < 3 pp and ΔCER < 2 pp (speaker bootstrap)."""
    rows = []
    for ch, g in asr.groupby("channel"):
        b = g[g.protection == "B0_unprotected"].set_index("utt")
        for p in g.protection.unique():
            if p == "B0_unprotected": continue
            q = g[g.protection == p].set_index("utt").join(b[["wer", "cer"]], rsuffix="0")
            q["dwer"] = (q.wer - q.wer0) * 100; q["dcer"] = (q.cer - q.cer0) * 100; q = q.reset_index()
            rng = np.random.default_rng(0); spk = q.speaker.unique(); bw, bc = [], []
            for _ in range(n_boot):
                s = rng.choice(spk, len(spk)); qq = pd.concat([q[q.speaker == k] for k in s]); bw.append(qq.dwer.mean()); bc.append(qq.dcer.mean())
            uw, uc = np.percentile(bw, 95), np.percentile(bc, 95)
            rows.append(dict(channel=ch, protection=p, WER_clean=q.wer0.mean() * 100, WER_prot=q.wer.mean() * 100, dWER_pp=q.dwer.mean(), dWER_upper95=uw,
                             dCER_pp=q.dcer.mean(), dCER_upper95=uc, H5_supported=bool(uw < 3 and uc < 2)))
    return pd.DataFrame(rows)
