#!/bin/sh
### General options
### –- specify queue --
#BSUB -q hpc

### -- set the job Name --
#BSUB -J filtering_language



### -- ask for number of cores (default: 1) --
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2000]" 

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 5:00


##BSUB -u s252263@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o data/jobs/filtering_%J.out
#BSUB -e data/jobs/filtering_%J.err
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
python3 filtering_language.py --input ~/ComputationalTools/AmazonReview/data/Books_rating_cleaned_new.csv 
