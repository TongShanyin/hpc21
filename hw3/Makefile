CXX=g++
CXXFLAGS=-std=c++11

all: omp-solved2 omp-solved3

omp-solved2: omp_solved2.c
	$(CXX) $(CXXFLAGS) -fopenmp omp_solved2.c -o omp-solved2

omp-solved3: omp_solved3.c
	$(CXX) $(CXXFLAGS) -fopenmp omp_solved3.c -o omp-solved3



clean:
	rm omp-solved2 omp-solved3
