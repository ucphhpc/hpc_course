# Need a bash shell below to get access to the compiler
SHELL := /bin/bash

LIB ?= -lm
INC ?= -I../include
FLAGS ?= -O3 -DNDEBUG -g -Wall

# Use multicore parallelization for OpenACC
FLAGS_ACC = -acc=multicore

# HDF5 Paths
H5_INC = /usr/include/hdf5/serial
H5_DIR = /usr/lib/x86_64-linux-gnu/hdf5/serial

# Flags for HDF5, should not be changed. Can be seen by running "h5c++ -show"
H5_FLAGS = -I$(H5_INC) -D_FORTIFY_SOURCE=2 -Wformat -Werror=format-security
H5 = -L$(H5_DIR) $(H5_DIR)/libhdf5_hl_cpp.a $(H5_DIR)/libhdf5_cpp.a $(H5_DIR)/libhdf5_hl.a $(H5_DIR)/libhdf5.a -Wl,-Bsymbolic-functions -Wl,-z,relro -lpthread -lsz -lz -ldl -lm -Wl,-rpath -Wl,$(H5_DIR) -Bstatic_pgi

MODULEPATH:=/opt/nvidia/hpc_sdk/modulefiles:$(MODULEPATH)

.PHONY: clean all

all: nbody_sequential

# Compile using Nvidia HPC compiler
nbody_sequential: nbody_sequential.cpp
	source /usr/share/modules/init/bash && \
        module load nvhpc && \
	nvc++ $(H5_FLAGS) -c $(FLAGS) $(INC) nbody_sequential.cpp
	source /usr/share/modules/init/bash && \
        module load nvhpc && \
	nvc++ $(H5_FLAGS) nbody_sequential.o $(FLAGS) $(INC) -o nbody_sequential $(LIB) $(H5)
	rm nbody_sequential.o

# Compile with OpenACC enabled
nbody_vectorised: nbody_vectorised.cpp
	source /usr/share/modules/init/bash && \
        module load nvhpc && \
	nvc++ $(H5_FLAGS) -c $(FLAGS_ACC) $(FLAGS) $(INC) nbody_vectorised.cpp
	source /usr/share/modules/init/bash && \
        module load nvhpc && \
	nvc++ $(H5_FLAGS) nbody_vectorised.o $(FLAGS_ACC) $(FLAGS) $(INC) -o nbody_vectorised $(LIB) $(H5)
	rm nbody_vectorised.o


clean:
	rm -f nbody_sequential nbody_vectorised *.o
