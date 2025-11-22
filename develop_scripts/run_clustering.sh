#!/bin/bash
#BSUB -q gpuv100
#BSUB -J test_UMAP
#BSUB -n 8
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -W 12:00


### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o data/jobs/clustering_%J.out
#BSUB -e data/jobs/clustering_%J.err

### -- Email notifications --
#BSUB -u laralechner@icloud.com
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
#module load python3/3.10.12
#module load gcc
#source ~/ComputationalTools/AmazonReview/.venv/bin/activate

# Activate conda env
module load cuda/11.6
source ~/miniforge3/bin/activate
conda activate text-embeddings 

# run filtering
python3 clustering_SBert.py 