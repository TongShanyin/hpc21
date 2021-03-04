CXX=g++
CXXFLAGS=-std=c++11 -O3 -march=native

HEADERS=utils.h intrin-wrapper.h

%.o: %.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) $(INC) $< -o $@

MMult1: MMult1.o 
	$(CXX) $(CXXFLAGS) MMult1.o -o MMult1

clean:
	rm *.o
