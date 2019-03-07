LIB ?= -lm -L/usr/lib -pthread -fopenmp -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lmpi_cxx
INC ?= -I../../include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include
FLAGS ?= -O3 -DNDEBUG -g -Wall

.PHONY: clean all

all: sw_sequential sw_parallel

sw_sequential: sw_sequential.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o sw_sequential sw_sequential.cpp

sw_parallel: sw_parallel.cpp
	h5c++ $(FLAGS) $(LIB) $(INC) -o sw_parallel sw_parallel.cpp

clean:
	rm -f sw_sequential sw_parallel *.o
