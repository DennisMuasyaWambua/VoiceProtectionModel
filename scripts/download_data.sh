#!/usr/bin/env bash
# Downloads the public corpora used (no login required). ~1.4 GB download.
set -euo pipefail
RAW=${1:-data/raw}; mkdir -p "$RAW" && cd "$RAW"
# LibriSpeech (CC BY 4.0) — OpenSLR SLR12
for s in dev-clean test-clean; do
  [ -d LibriSpeech/$s ] || { wget -c https://www.openslr.org/resources/12/$s.tar.gz && tar xzf $s.tar.gz && rm $s.tar.gz; }
done
# Optional, closer to the proposal (6.3 GB): train-clean-100
# wget -c https://www.openslr.org/resources/12/train-clean-100.tar.gz && tar xzf train-clean-100.tar.gz
# FLEURS Swahili — Kenya locale sw_ke (CC BY 4.0) — Google, via Hugging Face
mkdir -p fleurs_sw_ke && cd fleurs_sw_ke
for sp in test dev; do
  [ -f $sp.tsv ] || wget -c https://huggingface.co/datasets/google/fleurs/resolve/main/data/sw_ke/$sp.tsv
  [ -d $sp ] || { wget -c https://huggingface.co/datasets/google/fleurs/resolve/main/data/sw_ke/audio/$sp.tar.gz && tar xzf $sp.tar.gz && rm $sp.tar.gz; }
done
echo "done"
