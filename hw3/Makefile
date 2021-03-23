CXX=g++
CXXFLAGS=-std=c++11

all: omp-solved2 omp-solved3 omp-solved5

omp-solved2: omp_solved2.c
	$(CXX) $(CXXFLAGS) -fopenmp omp_solved2.c -o omp-solved2

omp-solved3: omp_solved3.c
	$(CXX) $(CXXFLAGS) -fopenmp omp_solved3.c -o omp-solved3


# omp-solved4: omp_solved4.c
# 	limit stacksize unlimited &&
# 	$(CXX) $(CXXFLAGS) -fopenmp omp_solved4.c -o omp-solved4

omp-solved5: omp_solved5.c
	$(CXX) $(CXXFLAGS) -fopenmp omp_solved5.c -o omp-solved5



clean:
	rm omp-solved2 omp-solved3 omp-solved5
