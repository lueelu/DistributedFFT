cmake_minimum_required(VERSION 3.10)
cmake_policy(VERSION 3.10)
project(HeffteTesting VERSION 2.1.0 LANGUAGES CXX)
enable_testing()

message(STATUS "heFFTe post-installation testing")

macro(heffte_add_mpi_test)
    cmake_parse_arguments(_heffte "" "NAME;COMMAND;RANKS" "" ${ARGN} )
    add_test(${_heffte_NAME} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_heffte_RANKS} ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/${_heffte_COMMAND} ${MPIEXEC_POSTFLAGS})
    unset(_heffte_NAME)
    unset(_heffte_RANKS)
    unset(_heffte_COMMAND)
endmacro()

find_package(Heffte 2.1.0 REQUIRED)

add_subdirectory("/home/lulu/binbin/heffte-new/heffte-lib/share/heffte/examples" examples)

if (Heffte_FFTW_FOUND)
    heffte_add_mpi_test(NAME example_fftw     COMMAND  examples/heffte_example_fftw     RANKS 2)
    heffte_add_mpi_test(NAME example_options  COMMAND  examples/heffte_example_options  RANKS 2)
    heffte_add_mpi_test(NAME example_vectors  COMMAND  examples/heffte_example_vectors  RANKS 2)
    heffte_add_mpi_test(NAME example_r2c      COMMAND  examples/heffte_example_r2c      RANKS 2)
endif()

if (Heffte_CUDA_FOUND)
    heffte_add_mpi_test(NAME example_cuda     COMMAND  examples/heffte_example_cuda     RANKS 2)
endif()

if (Heffte_FFTW_FOUND AND CMAKE_C_COMPILER)
    heffte_add_mpi_test(NAME example_c        COMMAND  examples/heffte_example_c        RANKS 2)
endif()

if (Heffte_FFTW_FOUND AND Heffte_Fortran_FOUND)
    heffte_add_mpi_test(NAME example_fortran  COMMAND  examples/heffte_example_fortran  RANKS 2)
endif()
