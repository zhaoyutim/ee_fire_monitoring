#!/usr/bin/env bash
#SBATCH --output /geoinfo_vol1/zhao2/run_logs/%J_slurm.out
#SBATCH --error  /geoinfo_vol1/zhao2/run_logs/%J_slurm.err
#SBATCH --mem  36GB
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 8

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /geoinfo_vol1/zhao2/mambaforge/etc/profile.d/conda.sh
conda activate myenv
PYTHONUNBUFFERED=1; python3 run_cnn_model.py -m $1 -p $2 -b $3 -bb $4
