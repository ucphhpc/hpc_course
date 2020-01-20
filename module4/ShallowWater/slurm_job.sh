#!/bin/bash

mpirun singularity exec ~/modi_images/hpc-notebook-latest.simg ./sw_parallel --iter 100 --size 1024
