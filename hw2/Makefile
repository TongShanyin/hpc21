CXX=g++
CXXFLAGS=-std=c++11

# HEADERS=utils.h intrin-wrapper.h

all: val_test01_solved val_test02_solved MMult1 fast-sin

val_test01_solved: val_test01_solved.cpp
	$(CXX) $(CXXFLAGS) val_test01_solved.cpp -o val_test01_solved


val_test02_solved: val_test02_solved.cpp
	$(CXX) $(CXXFLAGS) val_test02_solved.cpp -o val_test02_solved

MMult1: MMult1.cpp
	$(CXX) $(CXXFLAGS) -O3 -march=native -fopenmp MMult1.cpp -o MMult1

fast-sin: fast-sin.cpp
	$(CXX) $(CXXFLAGS) -O3 -march=native  fast-sin.cpp -o fast-sin


clean:
	rm val_test01_solved val_test02_solved MMult1 fast-sin

# %.o: %.cpp $(HEADERS)
# 	$(CXX) -c $(CXXFLAGS) $(INC) $< -o $@
#
# # MMult1: MMult1.o
# # 	$(CXX) $(CXXFLAGS) MMult1.o -o MMult1
#
# val_test01_solved: val_test01_solved.o
# 	$(CXX) $(CXXFLAGS) val_test01_solved.o -o val_test01_solved
#
