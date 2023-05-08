#include "hip/hip_runtime.h"
#include "../include/WMMA.h"
#include "../include/Utils.h"
#include <cstdio>
#include <cstdlib>


struct FFTMatrixHandle
{
    int Nx, Ny, N_batch;
    int radices_x[3] = {16, 16, 2};
    int radices_y[3] = {16, 16, 2};
    int n_radices_x, n_radices_y;
    int mergings[2] = {0, 0};
    void (*layer_0[3])(half2 *, half *, half *);
    void (*layer_1[3])(int, half2 *, half *, half *);
    half *F_real, *F_imag;
    half *F_real_tmp, *F_imag_tmp;
};

void FFTExec(FFTMatrixHandle plan, half *data);
void FFTCreate(FFTMatrixHandle *plan, int nx, int ny, int n_batch);


#define ErrChk(code) { Assert((code), __FILE__, __LINE__); }

static inline void Assert(hipError_t code, const char *file, int line) {
    if (code != hipSuccess) {
        printf("HIP Runtime Error: %s:%d:'%s'\n", file, line, hipGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}


#define KernelErrChk(){\
        hipError_t errSync  = hipGetLastError();\
        hipError_t errAsync = hipDeviceSynchronize();\
        if (errSync != hipSuccess) {\
              printf("Sync kernel error: %s\n", hipGetErrorString(errSync));\
              exit(EXIT_FAILURE);\
        }\
        if (errAsync != hipSuccess){\
            printf("Async kernel error: %s\n", hipGetErrorString(errAsync));\
            exit(EXIT_FAILURE);\
        }\
}
