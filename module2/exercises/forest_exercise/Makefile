LIB ?= -lbhxx -lboost_system
INC ?= -I./include -I../../../include
FLAGS ?= -Wall -Wno-narrowing


build:
	h5c++ $(FLAGS) $(LIB) $(INC) -o main src/util/*.cpp src/*.cpp main.cpp

clean:
	rm -fr main *.o

clean_steps:
	rm -fr forest_steps/
