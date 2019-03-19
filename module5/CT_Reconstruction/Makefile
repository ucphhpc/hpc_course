LIB ?= -lm -L/usr/lib -pthread -fopenmp
INC ?= -I../../include
FLAGS ?= -O3 -DNDEBUG -g -Wall

.PHONY: clean all

all: ct_sequential ct_parallel

ct_sequential: ct_sequential.cpp
	g++ $(FLAGS) $(LIB) $(INC) -o ct_sequential ct_sequential.cpp

ct_parallel: ct_parallel.cpp
	mpicxx $(FLAGS) $(LIB) $(INC) -o ct_parallel ct_parallel.cpp

clean:
	rm -f ct_sequential ct_parallel *.o
