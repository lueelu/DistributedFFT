OBJ = speed3d_c2c speed3d_r2c
INCLUDE = -I../../heffte/include -I../test -I/home/lulu/binbin/new-library/ompi/include -I/opt/rocm/rocfft/include 
LIB = -L../../heffte/lib -L/home/lulu/binbin/new-library/ompi/lib -L/opt/rocm/rocfft/lib
FLAGS = -std=c++14 -lmpi -lheffte -lrocfft -O3
ifdef DEBUG
FLAGS += -g -G
endif

all : $(OBJ)

speed3d_c2c : speed3d_c2c.cpp
	hipcc $^ -o $@  $(INCLUDE) $(LIB) $(FLAGS) 

speed3d_r2c : speed3d_r2c.cpp
	hipcc $^ -o $@  $(INCLUDE) $(LIB) $(FLAGS) 

.PHONY : clean

clean :
	rm -f $(OBJ)
