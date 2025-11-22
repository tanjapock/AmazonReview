#!/bin/bash
#BSUB -q gpuv100
#BSUB -J create_embeddings_SentenceBert
#BSUB -n 8
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -W 08:00
#BSUB -o data/jobs/embeddings_%J.out
#BSUB -e data/jobs/embeddings_%J.err


set -euo pipefail

ROOT=~/ComputationalTools/AmazonReview
IN="/dtu/blackhole/1a/222266/Book_rating_cleaned_engl_subset.csv"
OUT_DIR="/dtu/blackhole/1a/222266/embeddings_subset"   
LOG="$ROOT/data/logs/embeddings_${LSB_JOBID}.log"

module purge
module load python3/3.10.12
module load gcc
source "$ROOT/.venv/bin/activate"

mkdir -p "$ROOT/data/logs" "$OUT_DIR"


exec >"$LOG" 2>&1


python3 create_embeddings.py \
  --in "$IN" \
  --out "$OUT_DIR" \
  --chunksize 10000 \
  --batch-size 256 \
  --resume \
  --compression snappy \
  --log-file "$LOG"
