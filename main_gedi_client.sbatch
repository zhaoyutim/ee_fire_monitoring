#!/usr/bin/env bash
#SBATCH --output /geoinfo_vol1/zhao2/run_logs/%J_slurm.out
#SBATCH --error  /geoinfo_vol1/zhao2/run_logs/%J_slurm.err
#SBATCH --mem  100GB
#SBATCH --cpus-per-task 8

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
for i in {1..$1}
do
echo $i
PYTHONUNBUFFERED=1; python main_gedi_client.py -id $i
done
