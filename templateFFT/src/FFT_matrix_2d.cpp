#include "FFT_matrix_2d.h"
#include <cassert>
#include <hip/hip_fp16.h>
#include <string>



int *rev_x, *rev_y, Nx, Ny, N_batch;
float *in_host, *in_device_0;
FFTMatrixHandle plan;
  
void gen_rev(int N, int rev[], int radices[], int n_radices)
{
    int *tmp_0 = (int *)malloc(sizeof(int) * N);
    int *tmp_1 = (int *)malloc(sizeof(int) * N);
    int now_N = N;
// #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        tmp_0[i] = i;
    for (int i = n_radices - 1; i >= 0; --i)
    {
// #pragma omp parallel for
        for (int j = 0; j < N; j += now_N)
            for (int k = 0; k < radices[i]; ++k)
                for (int l = 0; l < now_N / radices[i]; ++l)
                {
                    tmp_1[j + l + k * (now_N / radices[i])] = tmp_0[j + l * radices[i] + k];
                }
        now_N /= radices[i];
        std::swap(tmp_0, tmp_1);
    }
// #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        rev[i] = tmp_0[i];
    // for(int i=0;i<N;i++){
    //     std::cout<<rev[i]<<" ";
    // }
    // std::cout<<std::endl;
}

void setup_2d(double *data, int nx, int ny, int n_batch)
{
    Nx = nx;
    Ny = ny;
    N_batch = n_batch;
    FFTCreate(&plan, Nx, Ny, N_batch);
   
 // in_host
    rev_x = (int *)malloc(sizeof(int) * Nx);
    rev_y = (int *)malloc(sizeof(int) * Ny);
    // std::cout<<"plan.n_radices_x:"<< plan.n_radices_x<<"  plan.n_radices_y:"<<plan.n_radices_y<< std::endl;
    // for(int i=0;i<3;i++){
    //     std::cout<<plan.radices_x[i]<<" ";
    // }
    // std::cout<<std::endl;
    // for(int i=0;i<3;i++){
    //     std::cout<<plan.radices_y[i]<<" ";
    // }
    // std::cout<<std::endl;
    
    gen_rev(Nx, rev_x, plan.radices_x, plan.n_radices_x);
    gen_rev(Ny, rev_y, plan.radices_y, plan.n_radices_y);
    in_host = (float *)malloc(sizeof(half) * 2 * Nx * Ny * N_batch);
    // for(int i=0;i<Nx;i++){
    //     std::cout<<rev_x[i]<<" ";
    // }
    // std::cout<<std::endl;
    //  for(int i=0;i<Ny;i++){
    //     std::cout<<rev_y[i]<<" ";
    // }
    // std::cout<<std::endl;
// #pragma omp parallel for
    for (int i = 0; i < N_batch; ++i)
        for (int j = 0; j < Nx; ++j)
            for (int k = 0; k < Ny; ++k)
            {
                //std::string s = std::to_string(data[2 * (i * Nx * Ny + rev_x[j] * Ny + rev_y[k]) + 0]);
                //float host;   
                //host = stof(s);
                //in_host[2 * (i * Nx * Ny + j * Ny + k) + 0] = __float2half(host);

                in_host[2 * (i * Nx * Ny + j * Ny + k) + 0] = data[2 * (i * Nx * Ny + rev_x[j] * Ny + rev_y[k]) + 0];

                //std::cout << std::fixed<< std::setprecision(10) << in_host[2 * (i * Nx * Ny + j * Ny + k) + 0] << ",";
                //std::cout <<  std::fixed<<std::setprecision(10) << data[2 * (i * Nx * Ny + rev_x[j] * Ny + rev_y[k]) + 0]<<"  ";

                in_host[2 * (i * Nx * Ny + j * Ny + k) + 1] = data[2 * (i * Nx * Ny + rev_x[j] * Ny + rev_y[k]) + 1];
            }
      
    ErrChk(hipMalloc(&in_device_0, sizeof(half) * Nx * Ny * N_batch * 2));
    ErrChk(hipMemcpy(in_device_0, in_host, sizeof(half) * Nx * Ny * N_batch * 2, hipMemcpyHostToDevice));

    #if 0
    for(int i = 0; i < nx; i++)
    {
        for(int j = 0; j < ny; j++)
        {
            std::cout << std::fixed<<"(" << std::setprecision(6)<<in_host[2 * (i * ny + j) + 0] <<","<<std::setprecision(6)<< in_host[2 * (i * ny + j) + 1] << ") ";
           
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    #endif
}

void finalize_2d(double *result)
{
    ErrChk(hipMemcpy(in_host, in_device_0, sizeof(float) * Nx * Ny * N_batch * 2, hipMemcpyDeviceToHost));
    for (int i = Nx * Ny - 8; i < Nx * Ny; i++) {
            std::cout << "element " << i
                 << " output: (" << in_host[2 * i] << "," << in_host[2 * i + 1] << ")" << std::endl;
    }
    for (int i = 0; i < N_batch * Nx * Ny; ++i)
    {
        result[2 * i + 0] = in_host[2 * i + 0];
        result[2 * i + 1] = in_host[2 * i + 1];
    }
}

void doit_2d()
{
    FFTExec(plan, in_device_0);
}