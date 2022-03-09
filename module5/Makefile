CXX := nvc++
GCC := g++

#######################################################
# Optimization flags are chosen as the last definition.
# Comment out using "#" at the begining of the line or rearrange according to your needs.
#
# Fastest executable (-ffast-math removes checking for NaNs and other things)
OPT=-O3 -fast 

# Add profiling to code
#OPT=-O1 -pg

# Faster compilation time
#OPT=-O1

INCLUDE := -I../include

CXXFLAGS := $(INCLUDE) $(OPT) -Wall -march=native -g -std=c++14

ACC := -acc=gpu -Minfo=acc


.PHONY: clean all

all: sw_sequential sw_parallel

sw_sequential: sw_sequential.cpp
	$(GCC) $(CXXFLAGS) sw_sequential.cpp -o sw_sequential

sw_parallel: sw_parallel.cpp
	$(CXX) $(CXXFLAGS) $(ACC) sw_parallel.cpp -o sw_parallel

clean:
	rm -f sw_sequential sw_parallel
