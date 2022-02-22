CXX := mpic++

#######################################################
# Optimization flags are chosen as the last definition.
# Comment out using "#" at the begining of the line or rearrange according to your needs.
#
# Fastest executable (-ffast-math removes checking for NaNs and other things)
OPT=-O3

CXXFLAGS := $(OPT) -Wall -march=native -g -std=c++14

default: task_farm

task_farm: task_farm.cpp
	$(CXX) task_farm.cpp $(CXXFLAGS) -o task_farm

clean:
	rm -f task_farm 
