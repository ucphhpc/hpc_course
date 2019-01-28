LIB ?= -lm -lbhxx -lboost_serialization -lboost_filesystem -lboost_system -lboost_regex
INC ?= -I/opt/hpc_course/include
FLAGS ?= -O3 -DNDEBUG -g

.PHONY: clean


nbody_seq: nbody_sequential.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o nbody_sequential nbody_sequential.cpp

clean:
	rm -f nbody_sequential *.o
