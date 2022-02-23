#!/usr/bin/env bash
#SBATCH --job-name=TF_8cores
#SBATCH --partition=modi_short
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive

mpiexec singularity exec \
   ~/modi_images/hpc-notebook_latest.sif \
   ./task_farm_HEP