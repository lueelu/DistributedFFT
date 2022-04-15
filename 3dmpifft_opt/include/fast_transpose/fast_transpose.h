#ifndef CUTRANSPOSE_H_
#define CUTRANSPOSE_H_

// #define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
// #cmakedefine USE_COMPLEX
// #cmakedefine FLOAT
// #define TILE_SIZE @CUT_TILE_SIZE@
// #define BRICK_SIZE @CUT_BRICK_SIZE@ 

#define TILE_SIZE 16
#define BRICK_SIZE 8
#define USE_COMPLEX 1


#ifdef USE_COMPLEX
#include <complex.h>
#endif

/********************************************
 * Public function prototypes               *
 ********************************************/
#ifdef __cplusplus
extern "C"
{
#endif

#ifdef FLOAT
#define real float
#else
#define real double
#endif


#ifdef USE_COMPLEX
typedef real _Complex data_t;
// typedef real data_t[2];
#else
typedef real data_t;
#endif

int cut_transpose3d( data_t*       output,
                     const data_t* input,
                     const int*    size,
                     const int*    permutation,
                     int           elements_per_thread );

#ifdef __cplusplus
}
#endif
#endif /* CUTRANSPOSE_H_ */
