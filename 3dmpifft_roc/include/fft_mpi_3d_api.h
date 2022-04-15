#ifndef __FFT_MPI_3D_API_H__
#define __FFT_MPI_3D_API_H__

#include "fft_mpi_common.h"
#include "kernel_func.h"
#include <fstream>
#include <vector>
#include "fast_transpose/fast_transpose.h"
#include "fast_transpose/kernels_102.h"

typedef struct fft_mpi_3d_plan
{
    hipfftHandle hipPlanX, hipPlanY, hipPlanZ, hipPlanYZ;
    rocfft_plan rocPlanYZ, rocPlanX;
    rocfft_execution_info rocPlanInfoYZ, rocPlanInfoX;
    size_t workBufSizeYZ, workBufSizeX;
    void *workBufYZ, *workBufX;
    int N[3];
    int locGPUIdx, devCountInNode, totalDevCount, globalDevIdx;
    int mpiRank, mpiSize;
    bool isLastDevice, isInplace;
    longInt64 maxDataCountInDevice;
    Complex *inDev, *outDev, *bufferDev1, *bufferDev2, **nodeDataDev;
    TransInfo tInfo;
    longInt64 lastExchangeN0, lastExchangeN1, lastExchangeN2;
    hipStream_t stream1, stream2;
    int direction;
#ifdef ENABLE_RCCL
    // init rccl
    ncclComm_t rccl_comm;
#endif

    fft_mpi_3d_plan() {
        hipPlanX = NULL;
        hipPlanY = NULL;
        hipPlanZ = NULL;
        hipPlanYZ = NULL;
        inDev = NULL;
        outDev = NULL;
        bufferDev1 = NULL;
        bufferDev2 = NULL;
        nodeDataDev = NULL;
        rocPlanYZ = NULL;
        rocPlanX = NULL;
        rocPlanInfoYZ = NULL;
        rocPlanInfoX = NULL;
        workBufYZ = NULL;
        workBufX = NULL;
        workBufSizeYZ = 0;
        workBufSizeX = 0;
    }
    
}*fft_mpi_3d_plan_p;

void fft_mpi_init(const longInt64 *N, int iniDeviceNumInNode, MPI_Comm comm, int &newDeviceCount, int &newDeviceCountInNode, longInt64 dataCountInNode[]);
void fft_mpi_cleanup(void);
fft_mpi_3d_plan_p fft_mpi_plan_dft_c2c_3d(longInt64 n0, longInt64 n1, longInt64 n2, Complex *in, Complex *out, Complex **node_data, MPI_Comm comm, int devIdx, int devCountInNode, int totalDevCount, int direction);
void fft_mpi_destroy_plan(fft_mpi_3d_plan_p plan);
void fft_mpi_execute_dft_3d_c2c(fft_mpi_3d_plan_p p);
longInt64 fft_mpi_local_size_3d(longInt64 n0, longInt64 n1, longInt64 n2, MPI_Comm comm, longInt64 *local_n0, longInt64 *local_0_start);
Complex* fft_mpi_alloc_local_memory(int count, int flag);


void getProperDeviceNum(const longInt64 *N, int iniDeviceNumInNode, int mpi_size, int mpi_rank, int &newDeviceCount, int &newDeviceCountInNode);
void getDataCountForNode(longInt64 dataCountInNode[], const longInt64 N[], int mpiRank, int mpiSize, int deviceCount, int deviceCountInNode);
longInt64 getMaxDataCount(int n0, int n1, int n2, int totalDevCount, bool isLastDevice);
void setFFTPlans(fft_mpi_3d_plan_p plan);
void outputPlanInfo(fft_mpi_3d_plan_p plan);
void fftZY(fft_mpi_3d_plan_p plan);
void fftX(fft_mpi_3d_plan_p plan);
void localTransposeUneven(fft_mpi_3d_plan_p plan);
void slabAlltoall(fft_mpi_3d_plan_p plan);
void debugLocalData(fft_mpi_3d_plan_p plan, int flag, int type);

#endif // __FFT_MPI_3D_API_H__