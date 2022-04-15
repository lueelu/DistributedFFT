cmake_minimum_required( VERSION 3.0 FATAL_ERROR )

project( distributedFFT )
 
############# Use message function to print information ############
message(STATUS "CMake version: ${CMAKE_VERSION}")
 
############ enable C++14 compiler###############
set(CMAKE_CXX_STANDARD 14)

# Add rocM root dir to CMAKE_PREFIX_PATH, usually /opt/rocm
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
find_package( hip REQUIRED )
find_package( rocfft REQUIRED )
find_package( rccl REQUIRED )


if(NOT WIN32)
  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -fopenmp")
set(CMAKE_CXX_FLAGS_Release "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ggdb -fopenmp")
 
ADD_DEFINITIONS(-DROCM=1)

# Add MPI dir to CMAKE_PREFIX_PATH
if(NOT DEFINED MPI_DIR)
  if(DEFINED ENV{MPI_DIR})
    set(MPI_DIR $ENV{MPI_DIR})
    list(APPEND CMAKE_PREFIX_PATH ${MPI_DIR})
    include_directories(${MPI_DIR}/include)
  endif()
else()
  list(APPEND CMAKE_PREFIX_PATH ${MPI_DIR})
  include_directories(${MPI_DIR}/include)
endif()
set( MPI_CXX_SKIP_MPICXX ON )
find_package( MPI REQUIRED )
 
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/speedTest.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
 
##########
##Build###
##########
add_executable(distFFTRoc fftSpeed3d_c2c.cpp
               include/fast_transpose/transpose3d.cpp 
               include/fast_transpose/kernels_012.cpp 
               include/fast_transpose/kernels_021.cpp 
               include/fast_transpose/kernels_102.cpp 
               include/fast_transpose/kernels_120.cpp 
               include/fast_transpose/kernels_201.cpp 
               include/fast_transpose/kernels_210.cpp
               include/fft_mpi_3d_api.cpp 
               include/kernel_func.cpp)

target_include_directories( distFFTRoc PUBLIC hip:device
               ${HIP_ROOT_DIR}/include
               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
               $<INSTALL_INTERFACE:include>)
           
target_link_libraries(distFFTRoc roc::rccl roc::rocfft ${MPI_CXX_LIBRARIES} )
install(TARGETS distFFTRoc RUNTIME DESTINATION bin)
