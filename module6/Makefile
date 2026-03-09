# instruct h5c++ to use mpic++ as the compiler and linker command
CXX = mpic++
export HDF5_CXX = $(CXX)
export HDF5_CXXLINKER = $(CXX)

# set compile flags
FLAGS = -O3 -DNDEBUG -g -Wall -march=native -std=c++23

.PHONY: clean all

all: domain_decomposition fwc_sequential fwc_parallel

domain_decomposition: domain_decomposition.cpp
	$(CXX) $(FLAGS) -o domain_decomposition domain_decomposition.cpp

fwc_sequential: fwc_sequential.cpp
	h5c++ $(FLAGS) -o fwc_sequential fwc_sequential.cpp

fwc_parallel: fwc_parallel.cpp
	h5c++ $(FLAGS) -o fwc_parallel fwc_parallel.cpp

clean:
	rm -f fwc_sequential fwc_parallel domain_decomposition *.o
