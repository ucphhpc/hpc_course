CXX := g++
CXXFLAGS := -g -std=c++14

MD: Water_sequential.o
	$(CXX) Water_sequential.o -o MD # Runs second ... with object files you only recompile changes

Water_sequential.o: Water_sequential.cpp
	$(CXX) -c Water_sequential.cpp $(CXXFLAGS) -o Water_sequential.o # Runs first

clean:
	rm -fr Water_sequential.o MD
