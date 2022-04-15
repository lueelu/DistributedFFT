#ifndef __FFT_MPI_COMMON_H__
#define __FFT_MPI_COMMON_H__

// #define __HIP_PLATFORM_HCC__

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <rocfft.h>
#include <hipfft.h>
#include <rccl.h>
#include <cmath>

#define ALLOC_CPU 1
#define ALLOC_DEV -1

#define FORWARD 1
#define BACKWARD -1

typedef double Complex[2];
typedef long long longInt64;

struct TransInfo {
    longInt64 *soffset;
    longInt64 *scount;
    longInt64 *roffset;
    longInt64 *rcount;
};

#define MPI_CHECK(stmt)                                          \
do {                                                             \
   int mpi_errno = (stmt);                                       \
   if (MPI_SUCCESS != mpi_errno) {                               \
       fprintf(stderr, "[%s:%d] MPI call failed with %d \n",     \
        __FILE__, __LINE__,mpi_errno);                           \
       exit(EXIT_FAILURE);                                       \
   }                                                             \
   assert(MPI_SUCCESS == mpi_errno);                             \
} while (0)

#define ROCM_CHECK(stmt)                                                \
do {                                                                    \
   hipError_t rocm_errno = (stmt);                                           \
   if (0 != rocm_errno) {                                                    \
       fprintf(stderr, "[%s:%d] ROCM call '%s' failed with %d: %s \n",  \
        __FILE__, __LINE__, #stmt, rocm_errno, hipGetErrorString(rocm_errno));    \
       exit(EXIT_FAILURE);                                              \
   }                                                                    \
   assert(hipSuccess == rocm_errno);                                         \
} while (0)

#define ROCFFT_CHECK(stmt)                                                \
do {                                                                    \
   rocfft_status rocm_errno = (stmt);                                           \
   if (rocfft_status_success != rocm_errno) {                                                    \
       fprintf(stderr, "RocFFT runtime error: %ud\n", rocm_errno);  \
       exit(EXIT_FAILURE);                                              \
   }                                                                    \
   assert(rocfft_status_success == rocm_errno);                                         \
} while (0)

#define HIPFFT_CHECK(stmt)                                                \
do {                                                                    \
   hipfftResult_t hipfft_errno = (stmt);                                           \
   if (HIPFFT_SUCCESS != hipfft_errno) {                                                    \
       fprintf(stderr, "HIPFFT runtime error: %ud\n", hipfft_errno);  \
       exit(EXIT_FAILURE);                                              \
   }                                                                    \
   assert(HIPFFT_SUCCESS == hipfft_errno);                                         \
} while (0)

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testNcclError = 3,
  testCuRandError = 4
} testResult_t;

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    gethostname(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                           \
  }                                                 \
} while(0)

#endif // __FFT_MPI_COMMON_H__