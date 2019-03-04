LIB ?= -lm -fopenmp
INC ?= -I../../include -I/opt/hpc_course/include
FLAGS ?= -O3 -DNDEBUG -g -Wall

.PHONY: clean all

all: fwc_sequential fwc_parallel

fwc_sequential: fwc_sequential.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o fwc_sequential fwc_sequential.cpp

fwc_parallel: fwc_parallel.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o fwc_parallel fwc_parallel.cpp

clean:
	rm -f fwc_sequential fwc_parallel *.o
