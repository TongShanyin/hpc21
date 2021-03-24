CXX=g++
CXXFLAGS=-std=c++11

all: omp-solved2 omp-solved3  omp-solved4 omp-solved5 omp-solved6 omp-scan jacobi2D-omp jacobi2D gs2D-omp gs2D

omp-solved2: omp_solved2.c
	$(CXX) $(CXXFLAGS) -fopenmp omp_solved2.c -o omp-solved2

omp-solved3: omp_solved3.c
	$(CXX) $(CXXFLAGS) -fopenmp omp_solved3.c -o omp-solved3

omp-solved4: omp_solved4.c
# 	limit stacksize unlimited &&
	$(CXX) $(CXXFLAGS) -fopenmp omp_solved4.c -o omp-solved4

omp-solved5: omp_solved5.c
	$(CXX) $(CXXFLAGS) -fopenmp omp_solved5.c -o omp-solved5

omp-solved6: omp_solved6.c
	$(CXX) $(CXXFLAGS) -fopenmp omp_solved6.c -o omp-solved6

omp-scan: omp-scan.cpp
		$(CXX) $(CXXFLAGS) -fopenmp omp-scan.cpp -o omp-scan

jacobi2D-omp: jacobi2D-omp.cpp
		$(CXX) $(CXXFLAGS) -fopenmp jacobi2D-omp.cpp -o jacobi2D-omp

jacobi2D: jacobi2D-omp.cpp utils.h
		$(CXX) $(CXXFLAGS) jacobi2D-omp.cpp -o jacobi2D

gs2D-omp: gs2D-omp.cpp
		$(CXX) $(CXXFLAGS) -fopenmp gs2D-omp.cpp -o gs2D-omp

gs2D: gs2D-omp.cpp utils.h
		$(CXX) $(CXXFLAGS) gs2D-omp.cpp -o gs2D


clean:
	rm omp-solved2 omp-solved3 omp-solved4 omp-solved5 omp-solved6 omp-scan jacobi2D-omp jacobi2D gs2D-omp gs2D
