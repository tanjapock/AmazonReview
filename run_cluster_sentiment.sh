#!/bin/bash
#BSUB -q hpc
#BSUB -J clustering_sentiment_umap

#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2000]" 

#BSUB -W 24:00


### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o data/new_logs/clustering_umap_%J.out
#BSUB -e data/new_logs/clustering_umap_%J.err

### -- Email notifications --
#BSUB -u tanja.pock@p2-it.de
#BSUB -N          # Notify at job completion
# -- end of LSF options --

set -euo pipefail

cd "$LS_SUBCWD"

# Debug info
echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "LS_SUBCWD: ${LS_SUBCWD:-unset}"
echo "Listing data/:"
ls -lah data || true


# Activate venv
module load python3/3.10.12
module load gcc
source ~/ComputationalTools/AmazonReview/.venv/bin/activate



# run filtering
python3 clustering_sentiment.py 