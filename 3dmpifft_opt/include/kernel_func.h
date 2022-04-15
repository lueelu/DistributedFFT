#ifndef __KERNEL_FUNC_H__
#define __KERNEL_FUNC_H__

// #define __HIP_PLATFORM_HCC__

#include <hip/hip_runtime.h>

void call_slab_local_transpose_z_to_x_uneven(double *data_in, double *data_out, int x_size, int y_size, int z_size, int y_divide, int y_last, int total_device, int direction);
void call_slab_local_transpose_z_to_x_uneven_optimized(double *data_in, double *data_out, int x_size, int y_size, int z_size, int y_divide, int y_last, int total_device, int direction);
void call_local_transpose_z_to_x_strided_forward(double *data_in, double *data_out, int x_size, int y_size, int z_size);
void call_local_transpose_z_to_x_strided_backward(double *data_in, double *data_out, int x_size, int y_size, int z_size);


__global__ void slab_local_transpose_z_to_x_uneven_forward(double *data_in, double *data_out, int y_divide, int y_last, int total_device);
__global__ void slab_local_transpose_z_to_x_uneven_backward(double *data_in, double *data_out, int y_divide, int y_last, int total_device);
__global__ void slab_local_transpose_z_to_x_uneven_forward_optimized(double *data_in, double *data_out, int y_divide, int y_last, int zSize, int total_device);
__global__ void slab_local_transpose_z_to_x_uneven_backward_optimized(double *data_in, double *data_out, int y_divide, int y_last, int total_device);

__global__ void local_transpose_z_to_x_strided_forward(double *data_in, double *data_out);
__global__ void local_transpose_z_to_x_strided_backward(double *data_in, double *data_out);

void call_local_transpose_z_to_x_strided_forward_opt(double *data_in, double *data_out, int x_size, int y_size, int z_size);
void call_local_transpose_z_to_x_strided_backward_opt(double *data_in, double *data_out, int x_size, int y_size, int z_size);
__global__ void local_transpose_z_to_x_strided_forward_opt(double *data_in, double *data_out);
__global__ void local_transpose_z_to_x_strided_backward_opt(double *data_in, double *data_out);

void call_scale_element(double *data, int N0, int N1, int N2, int x_remain);
__global__ void scale_element(double *data, int N0);

#endif // __KERNEL_FUNC_H__