"""Clean all corpora → data/clean/*.wav + data/manifests/{all,excluded}.csv (§3.3.5)."""
import os, sys, yaml, pandas as pd, soundfile as sf
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vpm.data import list_librispeech, list_fleurs, clean_one, speaker_split

cfg = yaml.safe_load(open("config.yaml")); A = cfg["audio"]; R = cfg["paths"]["raw"]; OUT = cfg["paths"]["clean"]

def work(row):
    y, info = clean_one(row["path"], A)
    if y is None: return {**row, "status": info}
    out = os.path.join(OUT, row["corpus"], row["utt_id"] + ".wav"); os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out, y, 16000, subtype="FLOAT")
    return {**row, **info, "status": "ok", "clean_path": out}

if __name__ == "__main__":
    dfs = [list_librispeech(R, "dev-clean"), list_librispeech(R, "test-clean"), list_fleurs(R, "test"), list_fleurs(R, "dev")]
    df = pd.concat(dfs, ignore_index=True)
    print("raw files:", len(df))
    with Pool(8) as p: res = p.map(work, df.to_dict("records"), chunksize=16)
    res = pd.DataFrame(res)
    os.makedirs(cfg["paths"]["manifests"], exist_ok=True)
    res[res.status != "ok"].to_csv(f"{cfg['paths']['manifests']}/excluded.csv", index=False)
    ok = res[res.status == "ok"].copy()
    ls_dev = speaker_split(ok[(ok.corpus == "librispeech") & (ok.subset == "dev-clean")], cfg["splits"]["n_train_speakers"], cfg["splits"]["n_val_speakers"], cfg["train"]["seed"])
    rest = ok.drop(ls_dev.index)
    rest["split"] = rest.apply(lambda r: "test" if r.corpus == "librispeech" else f"sw_{r.subset}", axis=1)
    allm = pd.concat([ls_dev, rest]).sort_index()
    allm.to_csv(f"{cfg['paths']['manifests']}/all.csv", index=False)
    print(allm.groupby(["corpus", "split"]).agg(n=("utt_id", "size"), spk=("speaker", "nunique"), hours=("dur", lambda d: d.sum() / 3600)))
    print(res.status.value_counts())
