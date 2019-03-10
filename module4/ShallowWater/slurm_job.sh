#!/bin/bash

mpirun shifter --image=nielsbohr/hpc-notebook ./sw_parallel --iter 100 --size 1024
