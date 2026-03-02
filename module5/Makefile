# CUDA C++ Compiler
NVCC := nvcc

#######################################################
# Optimization flags are chosen as the last definition.
# Comment out using "#" at the begining of the line or rearrange according to your needs.


# NVCC optimization flags
#NVCC_OPT=-O1 -G -g -lineinfo -dopt on
NVCC_OPT=-O3 --use_fast_math # generate optimised code

#NVCC_OPT += -pg     # Add profiling to code

# NVCC GPU tuning flags (may help, only do this if you know what it does!)
#NVCC_OPT += --maxrregcount=64 --extra-device-vectorization

NVCCFLAGS := $(NVCC_OPT) -std=c++20 -arch=native 

.PHONY: clean all

all: krakatau krakatau_cuda

# C++ version compiled with nvc++
krakatau: krakatau.cpp
	$(NVCC) $(NVCCFLAGS) krakatau.cpp -o krakatau

# CUDA version compiled with nvcc
krakatau_cuda: krakatau.cu
	$(NVCC) $(NVCCFLAGS) krakatau.cu -o krakatau_cuda

clean:
	rm -f krakatau krakatau_cuda

