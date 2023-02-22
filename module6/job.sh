#!/usr/bin/env bash
#SBATCH --job-name=FWC_8cores
#SBATCH --partition=modi_devel
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive

mpiexec singularity exec \
   ~/modi_images/hpc-notebook-latest.sif \
   ./fwc_parallel --iter 1000 --model models/small.hdf5
