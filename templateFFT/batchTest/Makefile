OBJ = Test_1D Test_2D rocFFT_1d rocFFT_2d
INCLUDE = -I ../src -I /opt/rocm/include -I/opt/rocm/rocfft/include
LIB = -L ../src -L/opt/rocm/lib -L/opt/rocm/rocfft/lib 
FLAGS = -D__HIP_PLATFORM_HCC__ -lrocfft -lamdhip64 -O3
ifdef DEBUG
FLAGS += -g -G
endif

all : $(OBJ)

Test_1D : Test_1D.cpp
	g++ $^ -o $@ ../src/templateFFT.cpp $(INCLUDE) $(LIB) $(FLAGS) 

Test_2D : Test_2D.cpp
	g++ $^ -o $@ ../src/templateFFT.cpp   $(INCLUDE) $(LIB) $(FLAGS) 

rocFFT_1d: rocFFT_1d.cpp
	hipcc $^ -o $@  $(INCLUDE) $(LIB) $(FLAGS) 

rocFFT_2d: rocFFT_2d.cpp
	hipcc $^ -o $@  $(INCLUDE) $(LIB) $(FLAGS) 

.PHONY : clean

clean :
	rm -f $(OBJ)
