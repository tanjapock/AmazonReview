#!/bin/bash
#BSUB -q gpuv100
#BSUB -J create_embeddings_SentenceBert
#BSUB -n 8
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -W 08:00
#BSUB -o /dev/null
#BSUB -e /dev/null

set -euo pipefail

ROOT=~/ComputationalTools/AmazonReview
IN="$ROOT/data/Books_rating_cleaned_only_eng.csv"
OUT_DIR="$ROOT/data/Books_rating_embeddings"   
LOG="$ROOT/data/logs/embeddings_${LSB_JOBID}.log"

module purge
module load python3/3.10.12
module load gcc
source "$ROOT/.venv/bin/activate"

mkdir -p "$ROOT/data/logs" "$OUT_DIR"


exec >"$LOG" 2>&1


python3 create_embeddings-1.py \
  --in "$IN" \
  --out "$OUT_DIR" \
  --chunksize 10000 \
  --batch-size 256 \
  --resume \
  --compression snappy \
  --log-file "$LOG"
