OBJ = speed3d_c2c speed3d_r2c
INCLUDE = -I../../include -I../test -I/<MPI_DIR>/ompi/include -I/opt/rocm/rocfft/include 
LIB = -L../../lib -L/<MPI_DIR>/ompi/lib -L/opt/rocm/rocfft/lib
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
