#!/bin/bash
# You can change the number of cores per node and the number of nodes here
# Total number of tasks:
#SBATCH -n 1
# Total number of nodes:
#SBATCH -N 1

mpirun singularity exec ~/modi_images/hpc-notebook-latest.sif ./sw_parallel --iter 100 --size 1024
