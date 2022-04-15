#include "fft_mpi_3d_api.h"
#include <unistd.h>
#include <omp.h>
#include <fstream>
#include <iostream>


int main(int argc, char* argv[]) {

    // convenient to debug 
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
	sleep(1);

    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided));
    if (provided != MPI_THREAD_SERIALIZED) {
        printf("could not support multi-thread MPI!\n");
        exit(EXIT_FAILURE);
    }

    int mpi_size, mpi_rank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

    if (argc != 5) {
        printf("The format of arguments should be [NX, NY, NZ, GPU_COUNT]!\n");
        exit(EXIT_FAILURE);
    }

    int devCount;
    ROCM_CHECK(hipGetDeviceCount(&devCount));

    const longInt64 N[3] = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
    const int iniDeviceNumInNode = atoi(argv[4]); 

    int newDeviceCount, newDeviceCountInNode;
    longInt64 dataCountInNode[iniDeviceNumInNode];

    fft_mpi_init(N, iniDeviceNumInNode, MPI_COMM_WORLD, newDeviceCount, newDeviceCountInNode, dataCountInNode);
    
    const int deviceCountInNode = newDeviceCountInNode, totalDeviceCount = newDeviceCount;

    // Alloc data on CPU
    Complex *data_cpu[deviceCountInNode], *data_cpu_out[deviceCountInNode], *node_data_dev[deviceCountInNode];
    double maxErrInProcess = 1e-30, maxErrTotal = 1e-30, forwardTimeProcess = 1e-30, forwardTimeTotal = 1e-30;
    #pragma omp parallel for num_threads(deviceCountInNode)
    {
        for (int i = 0; i < deviceCountInNode; ++i) {
            int globalIdx = mpi_rank * ceil((double)totalDeviceCount / mpi_size) + i;
            ROCM_CHECK(hipSetDevice(globalIdx % devCount));
            // ROCM_CHECK(hipSetDevice(i));
            // Assign values to input array
            int normalDeviceDataCount = ceil((double)N[0] / totalDeviceCount) * N[1] * N[2],
                normalDeviceCount = mpi_size==1? deviceCountInNode: (totalDeviceCount - deviceCountInNode) / (mpi_size - 1),
                normalNodeDataCount = normalDeviceCount * normalDeviceDataCount;
            data_cpu[i] = (Complex*)malloc(dataCountInNode[i] * sizeof(Complex));
            data_cpu_out[i] = (Complex*)malloc(dataCountInNode[i] * sizeof(Complex));
            for (int j = 0; j < dataCountInNode[i]; ++j) {
                data_cpu[i][j][0] = data_cpu[i][j][1] = mpi_rank * normalNodeDataCount + i * normalDeviceDataCount + j;
            }

            bool isLastDev = (mpi_rank * ceil((double)totalDeviceCount/mpi_size) + i) == (totalDeviceCount - 1);
            longInt64 maxDataCountDev = getMaxDataCount(N[0], N[1], N[2], totalDeviceCount, isLastDev);

            
            Complex *inDev, *outDev;
            inDev = fft_mpi_alloc_local_memory(maxDataCountDev, ALLOC_DEV);
            outDev = fft_mpi_alloc_local_memory(maxDataCountDev, ALLOC_DEV);
            ROCM_CHECK(hipMemcpyHtoD(inDev, data_cpu[i], dataCountInNode[i] * sizeof(Complex)));            

            fft_mpi_3d_plan_p plan = fft_mpi_plan_dft_c2c_3d(N[0], N[1], N[2], inDev, outDev, node_data_dev, MPI_COMM_WORLD, i, deviceCountInNode, totalDeviceCount, FORWARD);

            // outputPlanInfo(plan);
            // printf("forward and inverse FFT---------------------------------%d----------------------------------\n", globalIdx);
            ROCM_CHECK(hipMemcpyHtoD(plan->bufferDev1, data_cpu[i], dataCountInNode[i] * sizeof(Complex)));       
            fft_mpi_execute_dft_3d_c2c(plan);
            fft_mpi_3d_plan_p planBack = fft_mpi_plan_dft_c2c_3d(N[0], N[1], N[2], outDev, inDev, node_data_dev, MPI_COMM_WORLD, i, deviceCountInNode, totalDeviceCount, BACKWARD);
            // printf("inverse FFT---------------------------------%d----------------------------------\n", globalIdx);
            fft_mpi_execute_dft_3d_c2c(planBack);

            ROCM_CHECK(hipMemcpy(data_cpu_out[i], inDev, dataCountInNode[i] * sizeof(Complex), hipMemcpyDeviceToHost));
            double maxErr = -1.0;
            for (int j = 0; j < dataCountInNode[i]; ++j) {
                double tmp1 = data_cpu[i][j][0] - data_cpu_out[i][j][0] / (N[0] * N[1] * N[2]), tmp2 = data_cpu[i][j][1] - data_cpu_out[i][j][1] / (N[0] * N[1] * N[2]),
                        err = sqrt(std::pow(tmp1, 2) + std::pow(tmp2, 2))/1e7;
                if (maxErr < err)
                    maxErr = err;
            }

            // printf("performance---------------------------------%d----------------------------------\n", globalIdx);
            fft_mpi_execute_dft_3d_c2c(plan);
            double forward_time = -MPI_Wtime();
            fft_mpi_execute_dft_3d_c2c(plan);
            forward_time += MPI_Wtime() ;
            fft_mpi_execute_dft_3d_c2c(plan);


            fft_mpi_destroy_plan(plan);
            fft_mpi_destroy_plan(planBack);

            #pragma omp critical
            {
                if (maxErrInProcess < maxErr) {
                    maxErrInProcess = maxErr;
                }
                if (forwardTimeProcess < forward_time) {
                    forwardTimeProcess = forward_time;
                }
            }

            free(data_cpu_out[i]);
            free(data_cpu[i]);
            ROCM_CHECK(hipFree(inDev));
            ROCM_CHECK(hipFree(outDev));

            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        }
    }

    MPI_CHECK(MPI_Reduce(&maxErrInProcess, &maxErrTotal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Reduce(&forwardTimeProcess, &forwardTimeTotal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (mpi_rank == 0) {
        long long fftsize = N[0] * N[1] * N[2];
        double gflops = 5.0 * fftsize * std::log(fftsize) * 1e-9 / std::log(2.0) / forwardTimeTotal;
        std::cout << "\n----------------------------------------------------------------------------- \n";
        std::cout << "distributed FFT performance test\n";
        std::cout << "----------------------------------------------------------------------------- \n";
        std::cout << "Size:             " << N[0] << "x" << N[1] << "x" << N[2] << "\n";
        std::cout << "MPI ranks:        " << mpi_size << "\n";
        std::cout << "Forward FFT time: " << forwardTimeTotal << " (s)\n";
        std::cout << "Performance:      " << gflops << " GFlops/s\n";
        std::cout << "Max error:        " << maxErrTotal << "\n";
        std::cout << std::endl;
    }

    MPI_CHECK(MPI_Finalize());

    return 0;
}