#
# Benchmarks
#

macro(heffte_add_benchmark heffte_benchmark)
    find_package(heffte PATHS /home/lulu/binbin/heffte-new/heffte-lib)
    add_executable(${heffte_benchmark} "${heffte_benchmark}.cpp")
    target_link_libraries(${heffte_benchmark} heffte)

    # using some of the helper methods from the test-modules
    target_include_directories(${heffte_benchmark} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../test/> PUBLIC /home/lulu/binbin/heffte-new/heffte-lib/include PUBLIC /opt/rocm/include)
endmacro()

heffte_add_benchmark(speed3d_c2c)
heffte_add_benchmark(speed3d_r2c)
