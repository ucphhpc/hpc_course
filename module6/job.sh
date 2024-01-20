#!/usr/bin/env bash
#SBATCH --job-name=FWC_8cores
#SBATCH --partition=modi_HPPC
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive

mpiexec apptainer exec \
   ~/modi_images/ucphhpc/hpc-notebook:latest \
   ./fwc_parallel --iter 1000 --model models/small.hdf5
