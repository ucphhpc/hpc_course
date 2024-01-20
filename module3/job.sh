#!/usr/bin/env bash
#SBATCH --job-name=TF_8cores
#SBATCH --partition=modi_devel
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive

mpiexec apptainer exec \
   ~/modi_images/ucphhpc/hpc-notebook:latest \
   ./task_farm_HEP