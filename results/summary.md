## 9. Summary of real results (read with the caveats in §10)
*Evaluation is **partial**: run_eval.py was checkpointed after 30 of the 80 planned references (15 of the 40 test speakers) when the deliverable was cut. Re-run `scripts/run_eval.py` for the full set.*

* **Seen verifier (ECAPA, E1, direct):** the B6 generator (G-M, 10 ms look-ahead) cuts acceptance from 1.00 to 0.70 (ΔAR +0.30, 95 % CI +0.10 to +0.50). After a simulated T_KE channel it goes from 0.60 to 0.37. The PGD-EOT teacher reaches AR 0.00 on the seen encoder.
* **Black-box transfer:** the effect transfers strongly to **Resemblyzer** (direct: AR 1.00 → 0.17; after Opus 48k: 0.83 → 0.20), but **almost not at all to WavLM-SV** (direct: 0.93 → 0.93; after G.711: 0.83 → 0.67). The teacher transfers to WavLM only partly (direct 0.93 → 0.60). So the transfer claim (H4) is **not supported** for the strongest held-out verifier at this scale.
* **H2 proxy (B6 T_KE vs B5 generic, ECAPA):** results are mixed and the CIs overlap, so there is no evidence either way at n = 15 speakers.
* **G-S (0 ms look-ahead):** it collapsed to a near-silent perturbation (SNR ≈ 55 dB, ΔAR ≈ 0) after 800 steps.
* **Quality:** B6 has SNR 20.8 dB, PESQ-WB 2.03 and STOI 0.97. That is clearly audible and well below the proposal's ViSQOL ≥ 3.8 target, so ε = 0.008 / λ = 0.1 is **too aggressive**. With λ = 1 the generator learned no protection at all.
* **H5 (ASR, Whisper base.en):** ΔWER +0.36 pp direct (upper 95 % 0.83) and +1.65 pp after G.711 (upper 2.34 < 3). H5 non-inferiority is supported.
* **Swahili (FLEURS sw_ke):** utility is similar to English (PESQ 2.48, STOI 0.93), and the ECAPA displacement is similar (cos 0.76 vs 0.70).
* **ONNX:** FP32 matches PyTorch to 6e-9. Naive INT8 had 26 % relative error. Sensitivity-aware INT8 (dynamic 0.8 %, static QDQ 1.3 %) passes the 1e-3 check and keeps protection (ρ_INT8 ≈ ρ_FP32).
