# Voice Protection Model — causal streaming adversarial perturbations against voice cloning
Dennis — MSc Data Science & Analytics, Strathmore University. A CPU-scale research prototype of Chapter 3 of the proposal (`latex_v3`).

## What's included
```
notebooks/Voice_Protection_Model.ipynb   executed notebook: data → EDA → preprocessing → training curves → evaluation → ONNX → INT8 → benchmark
src/vpm/          data.py (cleaning, splits) · channels.py (T_KE, held-out codecs, removal attacks) · losses.py (MR-STFT + masking, Eqs 5–7)
                  generator.py (causal TCN G-S / G-M + GRU, ε·tanh, look-ahead, streaming step) · encoders.py (ECAPA seen; WavLM-SV, Resemblyzer held-out)
                  train.py (Eq. 11, curriculum, mode switch) · teacher.py (PGD-EOT, Eqs 3/8/9) · removal.py (E2 removal net)
                  evaluate.py / report.py (EER thresholds, AR, ΔAR with speaker bootstrap, ρ, H5) · export.py (ONNX, ORT INT8, benchmark) · eda.py
scripts/          download_data.sh · prepare_data.py · run_eval.py · run_utility.py · run_e2.py · train_removal.py · export_all.py · build_notebook.py
tests/test_streaming.py   streaming == offline unit test (§3.6.2)
models/           gm_ke_lam0.1 (B6 main, G-M, 10 ms look-ahead) · gm_gen_lam0.1 (B5) · gs_ke_lam0.1 (G-S, 0 ms) · gm_ke_lam1 (log only)
                  removal_net_E2.pt (white-box attacker – research only, do not release) · onnx/gm_ke/*.onnx (fp32, ORT-optimised, INT8 dynamic, INT8 static QDQ)
results/, plots/  all CSVs and figures used in the notebook
config.yaml, requirements.txt
```

## Reproduce
```bash
python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt   # needs ffmpeg (libopus, libmp3lame) and python3-dev
bash scripts/download_data.sh data/raw                                            # LibriSpeech dev/test-clean + FLEURS sw_ke (~1.4 GB)
python scripts/prepare_data.py
export PYTHONPATH=src
python tests/test_streaming.py
python -m vpm.train --variant GM --tdist ke  --lam 0.1 --steps 800 --out models/gm_ke_lam0.1   # ~45 min on 4 CPU threads
python -m vpm.train --variant GM --tdist gen --lam 0.1 --steps 800 --out models/gm_gen_lam0.1
python -m vpm.train --variant GS --tdist ke  --lam 0.1 --steps 800 --out models/gs_ke_lam0.1
python scripts/train_removal.py models/gm_ke_lam0.1/best.pt models/removal_net_E2.pt
python scripts/export_all.py models/gm_ke_lam0.1/best.pt models/onnx/gm_ke --bench
python scripts/run_eval.py --main models/gm_ke_lam0.1/best.pt --gen_baseline models/gm_gen_lam0.1/best.pt \
       --gs models/gs_ke_lam0.1/best.pt --int8 models/onnx/gm_ke/step_int8_dynamic.onnx
python scripts/run_e2.py; python scripts/run_utility.py models/gm_ke_lam0.1/best.pt models/onnx/gm_ke/step_int8_dynamic.onnx
python scripts/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/Voice_Protection_Model.ipynb
```
Streaming use: feed `models/onnx/gm_ke/step_int8_dynamic.onnx` one 160-sample block at a time with inputs `block, prev, mode(one-hot mobile/pa_live/unknown), conv_state (1,96,S), gru_state (1,1,96)`; see `vpm.export.ort_stream`.

## Results
See `RESULTS.md` (auto-written from the result CSVs) and the notebook.
