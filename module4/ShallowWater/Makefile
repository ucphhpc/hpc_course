LIB ?= -lm -fopenmp
INC ?= -I../../include
FLAGS ?= -O3 -DNDEBUG -g -Wall

.PHONY: clean all

all: sw_sequential

sw_sequential: sw_sequential.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o sw_sequential sw_sequential.cpp

clean:
	rm -f sw_sequential *.o
