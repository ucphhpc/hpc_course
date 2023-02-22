#!/usr/bin/env bash
#SBATCH --job-name=Seismogram
#SBATCH --partition=modi_devel
#SBATCH --nodes=1 --ntasks=1 --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

# set loop scheduling to static
export OMP_SCHEDULE=static

# Schedule one thread per core. Change to "threads" for hyperthreading
export OMP_PLACES=cores
#export OMP_PLACES=threads

# Place threads as close to each other as possible
export OMP_PROC_BIND=close

# Set and print number of cores / threads to use
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo Number of threads=$OMP_NUM_THREADS

# uncomment to write info about binding and environment variables to screen
#export OMP_DISPLAY_ENV=true

singularity exec ~/modi_images/hpc-notebook-latest.sif ./mp
