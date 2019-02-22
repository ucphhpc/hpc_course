LIB ?= -lm -lbhxx -lboost_serialization -lboost_filesystem -lboost_system -lboost_regex
INC ?= -I../../include
FLAGS ?= -O3 -DNDEBUG -g -Wall

.PHONY: clean all

all: nbody_sequential nbody_bhxx

nbody_sequential: nbody_sequential.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o nbody_sequential nbody_sequential.cpp

nbody_bhxx: nbody_bhxx.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o nbody_bhxx nbody_bhxx.cpp

clean:
	rm -f nbody_sequential nbody_bhxx *.o
