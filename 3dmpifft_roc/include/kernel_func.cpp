#include "kernel_func.h"

void call_slab_local_transpose_z_to_x_uneven(double *data_in, double *data_out, int x_size, int y_size, int z_size, int y_divide, int y_last, int total_device, int direction) 
{
    dim3 grid_size(x_size, y_size, z_size), block_size;
    if (direction == 1)
        hipLaunchKernelGGL(slab_local_transpose_z_to_x_uneven_forward, grid_size, block_size, 0, 0, data_in, data_out, y_divide, y_last, total_device);
    else if (direction == -1)
        hipLaunchKernelGGL(slab_local_transpose_z_to_x_uneven_backward, grid_size, block_size, 0, 0, data_in, data_out, y_divide, y_last, total_device);
}

void call_slab_local_transpose_z_to_x_uneven_optimized(double *data_in, double *data_out, int x_size, int y_size, int z_size, int y_divide, int y_last, int total_device, int direction) 
{
    dim3 grid_size(x_size, y_size, 1), block_size(z_size);
    if (direction == 1)
        hipLaunchKernelGGL(slab_local_transpose_z_to_x_uneven_forward_optimized, grid_size, block_size, 0, 0, data_in, data_out, y_divide, y_last, total_device);
    else if (direction == -1)
        hipLaunchKernelGGL(slab_local_transpose_z_to_x_uneven_backward_optimized, grid_size, block_size, 0, 0, data_in, data_out, y_divide, y_last, total_device);
}

void call_local_transpose_z_to_x_strided_backward(double *data_in, double *data_out, int x_size, int y_size, int z_size) 
{
    dim3 grid_size(x_size, y_size, z_size), block_size(1);
    hipLaunchKernelGGL(local_transpose_z_to_x_strided_backward, grid_size, block_size, 0, 0, data_in, data_out);
}

void call_local_transpose_z_to_x_strided_forward(double *data_in, double *data_out, int x_size, int y_size, int z_size) 
{
    dim3 grid_size(x_size, y_size, z_size), block_size(1);
    hipLaunchKernelGGL(local_transpose_z_to_x_strided_forward, grid_size, block_size, 0, 0, data_in, data_out);
}

__global__ void slab_local_transpose_z_to_x_uneven_forward(double *data_in, double *data_out, int y_divide, int y_last, int total_device) 
{
    int x_size = gridDim.x, y_size = gridDim.y, z_size = gridDim.z;
    int idx1 = (blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z) * 2, idx2;
    // int idx2 = (x_size * y_divide * z_size * (blockIdx.y / gridDim.y) + blockIdx.x * y_divide * gridDim.z + blockIdx.x % y_divide * gridDim.z + blockIdx.z) * 2;
    if (blockIdx.y / y_divide == total_device - 1)  
        idx2 = (x_size * y_divide * z_size * (blockIdx.y / y_divide) + blockIdx.x * y_last * gridDim.z + blockIdx.y % y_divide * gridDim.z + blockIdx.z) * 2;
    else
        idx2 = (x_size * y_divide * z_size * (blockIdx.y / y_divide) + blockIdx.x * y_divide * gridDim.z + blockIdx.y % y_divide * gridDim.z + blockIdx.z) * 2;
    // printf("(%d->%d)", idx1/2, idx2/2);
    data_out[idx2] = data_in[idx1];
    data_out[idx2 + 1] = data_in[idx1 + 1];
}

__global__ void slab_local_transpose_z_to_x_uneven_backward(double *data_in, double *data_out, int y_divide, int y_last, int total_device) 
{
    int x_size = gridDim.x, y_size = gridDim.y, z_size = gridDim.z;
    int idx1 = (blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z) * 2, idx2;
    // int idx2 = (x_size * y_divide * z_size * (blockIdx.y / gridDim.y) + blockIdx.x * y_divide * gridDim.z + blockIdx.x % y_divide * gridDim.z + blockIdx.z) * 2;
    if (blockIdx.y / y_divide == total_device - 1)  
        idx2 = (x_size * y_divide * z_size * (blockIdx.y / y_divide) + blockIdx.x * y_last * gridDim.z + blockIdx.y % y_divide * gridDim.z + blockIdx.z) * 2;
    else
        idx2 = (x_size * y_divide * z_size * (blockIdx.y / y_divide) + blockIdx.x * y_divide * gridDim.z + blockIdx.y % y_divide * gridDim.z + blockIdx.z) * 2;
    // printf("(%d->%d)", idx1/2, idx2/2);
    data_out[idx1] = data_in[idx2];
    data_out[idx1 + 1] = data_in[idx2 + 1];
}

__global__ void slab_local_transpose_z_to_x_uneven_forward_optimized(double *data_in, double *data_out, int y_divide, int y_last, int total_device) 
{
    int x_size = gridDim.x, y_size = gridDim.y, z_size = blockDim.x;
    int idx1 = (blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x) * 2, idx2;
    // int idx2 = (x_size * y_divide * z_size * (blockIdx.y / gridDim.y) + blockIdx.x * y_divide * gridDim.z + blockIdx.x % y_divide * gridDim.z + blockIdx.z) * 2;
    if (blockIdx.y / y_divide == total_device - 1)  
        idx2 = (x_size * y_divide * z_size * (blockIdx.y / y_divide) + blockIdx.x * y_last * blockDim.x + blockIdx.y % y_divide * blockDim.x + threadIdx.x) * 2;
    else
        idx2 = (x_size * y_divide * z_size * (blockIdx.y / y_divide) + blockIdx.x * y_divide * blockDim.x + blockIdx.y % y_divide * blockDim.x + threadIdx.x) * 2;
    // printf("(%d->%d)", idx1/2, idx2/2);
    data_out[idx2] = data_in[idx1];
    data_out[idx2 + 1] = data_in[idx1 + 1];
}

__global__ void slab_local_transpose_z_to_x_uneven_backward_optimized(double *data_in, double *data_out, int y_divide, int y_last, int total_device) 
{
    int x_size = gridDim.x, y_size = gridDim.y, z_size = blockDim.x;
    int idx1 = (blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x) * 2, idx2;
    // int idx2 = (x_size * y_divide * z_size * (blockIdx.y / gridDim.y) + blockIdx.x * y_divide * gridDim.z + blockIdx.x % y_divide * gridDim.z + blockIdx.z) * 2;
    if (blockIdx.y / y_divide == total_device - 1)  
        idx2 = (x_size * y_divide * z_size * (blockIdx.y / y_divide) + blockIdx.x * y_last * blockDim.x + blockIdx.y % y_divide * blockDim.x + threadIdx.x) * 2;
    else
        idx2 = (x_size * y_divide * z_size * (blockIdx.y / y_divide) + blockIdx.x * y_divide * blockDim.x + blockIdx.y % y_divide * blockDim.x + threadIdx.x) * 2;
    // printf("(%d->%d)", idx1/2, idx2/2);
    data_out[idx1] = data_in[idx2];
    data_out[idx1 + 1] = data_in[idx2 + 1];
}

__global__ void local_transpose_z_to_x_strided_forward(double *data_in, double *data_out) 
{
    int idx1 = (blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z) * 2,
        idx2 = (blockIdx.z * gridDim.x + blockIdx.y * gridDim.x * gridDim.z + blockIdx.x) * 2;
    data_out[idx2] = data_in[idx1];
    data_out[idx2 + 1] = data_in[idx1 + 1];
}

__global__ void local_transpose_z_to_x_strided_backward(double *data_in, double *data_out) 
{
    int idx1 = (blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z) * 2,
        idx2 = (blockIdx.z * gridDim.x + blockIdx.y * gridDim.x * gridDim.z + blockIdx.x) * 2;
    data_out[idx1] = data_in[idx2];
    data_out[idx1 + 1] = data_in[idx2 + 1];
}

void call_local_transpose_z_to_x_strided_forward_opt(double *data_in, double *data_out, int x_size, int y_size, int z_size) 
{
     dim3 grid_size(x_size, y_size, 1), block_size(z_size, 1, 1);
    hipLaunchKernelGGL(local_transpose_z_to_x_strided_forward_opt, grid_size, block_size, 0, 0, data_in, data_out);
}

void call_local_transpose_z_to_x_strided_backward_opt(double *data_in, double *data_out, int x_size, int y_size, int z_size) 
{
    dim3 grid_size(x_size, y_size, 1), block_size(z_size, 1, 1);
    hipLaunchKernelGGL(local_transpose_z_to_x_strided_backward_opt, grid_size, block_size, 0, 0, data_in, data_out);
}

__global__ void local_transpose_z_to_x_strided_forward_opt(double *data_in, double *data_out) 
{
    int idx1 = (blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x) * 2,
        idx2 = (threadIdx.x * gridDim.x + blockIdx.y * gridDim.x * blockDim.x + blockIdx.x) * 2;
    data_out[idx2] = data_in[idx1];
    data_out[idx2 + 1] = data_in[idx1 + 1];
}

__global__ void local_transpose_z_to_x_strided_backward_opt(double *data_in, double *data_out) 
{
    int idx1 = (blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x) * 2,
        idx2 = (threadIdx.x * gridDim.x + blockIdx.y * gridDim.x * blockDim.x + blockIdx.x) * 2;
    data_out[idx1] = data_in[idx2];
    data_out[idx1 + 1] = data_in[idx2 + 1];
}

void call_scale_element(double *data, int N0, int N1, int N2, int x_remain) 
{
    dim3 grid_size(x_remain, N1, 1), block_size(N2, 1, 1);
    hipLaunchKernelGGL(scale_element, grid_size, block_size, 0, 0, data, N0);
}

__global__ void scale_element(double *data, int N0) 
{
    int idx = (blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x) * 2;

    data[idx] /= (double)N0 * gridDim.y * blockDim.x;
    data[idx + 1] /= (double)N0 * gridDim.y * blockDim.x;
}