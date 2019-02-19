LIB ?= -lm -fopenmp
INC ?= -I../../include
FLAGS ?= -O3 -DNDEBUG -g -Wall

.PHONY: clean all

all: fwc_sequential

fwc_sequential: fwc_sequential.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o fwc_sequential fwc_sequential.cpp

clean:
	rm -f fwc_sequential *.o
