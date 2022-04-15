// Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <cassert>
#include <complex>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <hip/hip_runtime_api.h>

#include "rocfft.h"

int main(int argc, char* argv[])
{
    std::cout << "rocFFT complex 2d FFT example\n";

    // The problem size
    const size_t Nx      = (argc < 2) ? 8 : atoi(argv[1]);
    const size_t Ny      = (argc < 3) ? 8 : atoi(argv[2]);
    const bool   inplace = (argc < 4) ? false : atoi(argv[3]);

    size_t nbatch = (uint64_t)(64 * 32 * (uint64_t)pow(2, 15))/(Nx * Ny);
    std::cout << "Nx: " << Nx << "\tNy: " << Ny << "\tbatch: "<< nbatch<<"\tin-place: " << inplace << std::endl;

    //Initialize data on the host
    // std::cout << "Input:\n";
    std::vector<std::complex<double>> cx(Nx * Ny * nbatch);
    for(size_t i = 0; i < Nx * Ny * nbatch; i++)
    {
        cx[i] = std::complex<double>(i, 0);
    }

    hipSetDevice(3);
    rocfft_setup();

    // Create HIP device object and copy data:
    double2* x = NULL;
    hipMalloc(&x, cx.size() * sizeof(decltype(cx)::value_type));
    uint64_t bufferSize = cx.size() * sizeof(decltype(cx)::value_type);
    double2* y = inplace ? (double2*)x : NULL;
    if(!inplace)
    {
        hipMalloc(&y, cx.size() * sizeof(decltype(cx)::value_type));
    }
    hipMemcpy(x, cx.data(), cx.size() * sizeof(decltype(cx)::value_type), hipMemcpyHostToDevice);

    // double *x = NULL;
    // double* y = inplace ? (double*)x : NULL;
    // if(!inplace)
    // {
    //     hipMalloc(&y, sizeof(double) * Nx * Ny * 2);
    // }
    // hipMalloc(&x, sizeof(double) * Nx * Ny * 2);
    // hipMemcpy(x, data, sizeof(double) * Nx * Ny * 2, hipMemcpyHostToDevice);



    // Length are in reverse order because rocfft is column-major.
    const size_t lengths[2] = {Ny, Nx};

    rocfft_status status = rocfft_status_success;

    // Create plans
    rocfft_plan forward = NULL;
    status              = rocfft_plan_create(&forward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_complex_forward,
                                rocfft_precision_double,
                                2, // Dimensions
                                lengths, // lengths
                                nbatch, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    // We may need work memory, which is passed via rocfft_execution_info
    rocfft_execution_info forwardinfo = NULL;
    status                            = rocfft_execution_info_create(&forwardinfo);
    assert(status == rocfft_status_success);
    size_t fbuffersize = 0;
    status             = rocfft_plan_get_work_buffer_size(forward, &fbuffersize);
    assert(status == rocfft_status_success);
    void* fbuffer = NULL;
    if(fbuffersize > 0)
    {
        hipMalloc(&fbuffer, fbuffersize);
        status = rocfft_execution_info_set_work_buffer(forwardinfo, fbuffer, fbuffersize);
        assert(status == rocfft_status_success);
    }

    // Create plans
    rocfft_plan backward = NULL;
    status               = rocfft_plan_create(&backward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_complex_inverse,
                                rocfft_precision_double,
                                2, // Dimensions
                                lengths, // lengths
                                nbatch, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    // Execution info for the backward transform:
    rocfft_execution_info backwardinfo = NULL;
    status                             = rocfft_execution_info_create(&backwardinfo);
    assert(status == rocfft_status_success);
    size_t bbuffersize = 0;
    status             = rocfft_plan_get_work_buffer_size(backward, &bbuffersize);
    assert(status == rocfft_status_success);
    void* bbuffer = NULL;
    if(bbuffersize > 0)
    {
        hipMalloc(&bbuffer, bbuffersize);
        status = rocfft_execution_info_set_work_buffer(backwardinfo, bbuffer, bbuffersize);
        assert(status == rocfft_status_success);
    }

    // Execute the forward transform
    status = rocfft_execute(forward,
                            (void**)&x, // in_buffer
                            (void**)&y, // out_buffer
                            forwardinfo); // execution info
    assert(status == rocfft_status_success);



    hipEvent_t start, stop;
    float elapsedTime = 0.f;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, NULL);

    for(int i = 0; i < 1000; i++){
        status = rocfft_execute(forward,
                            (void**)&x, // in_buffer
                            (void**)&y, // out_buffer
                            forwardinfo); // execution info
        assert(status == rocfft_status_success);

    }

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);

    // Copy result back to host
    std::vector<std::complex<double>> cy(cx.size());
    hipMemcpy(cy.data(), y, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);

	double opscount = (double)5 * nbatch * Nx * Ny * log(Nx * Ny) / log(2.0);

    printf("nx: %zu, ny: %zu, iter times: %d,  total time: %lf, time per iter: %lfms\n", Nx, Ny, 1000, elapsedTime, elapsedTime/1000);

    


    // Execute the backward transform
    rocfft_execute(backward,
                   (void**)&y, // in_buffer
                   (void**)&x, // out_buffer
                   backwardinfo); // execution info

    // std::cout << "Transformed back:\n";
    hipMemcpy(cy.data(), x, cy.size() * sizeof(decltype(cy)::value_type), hipMemcpyDeviceToHost);


    const double overN = 1.0f /(Nx * Ny);
    
    double tmp1, tmp2, tmp3 ,maxErr = 0.0;
	for (int i = 0; i < cx.size(); i++) {
		tmp1 = cx[i].real()-cy[i].real() * overN;
        tmp1 = cx[i].imag()-cy[i].imag() * overN;	
        tmp3 = sqrt( tmp1*tmp1 + tmp2*tmp2 );
		maxErr = maxErr >= tmp3 ? maxErr : tmp3;

	}
    std::cout << "Maximum error: " << maxErr << "\n";

    #if 1
	char filename[30] = "batch_rocResult2D.csv";
	std::ofstream outFile;
	outFile.open(filename, std::ios::app);
	//outFile<<"X"<<','<<"Y"<<','<<"Z"<<','<<"Buffer"<<','<<"hip_time"<<','<<"GFlops"<<','<<"num_iter"<<','<<"bandwidth"<<','<<"max error"<<std::endl;
	outFile<<Nx<<','<<Ny<<','<<1<<','<<bufferSize / 1024.0 / 1024.0<<','<<elapsedTime/1000<<','<<opscount/(1e6 * elapsedTime / 1000)<<','<<1000<<','<<bufferSize / 1024.0 / 1024.0 /1.024 * 2/ (elapsedTime/1000)<<','<<maxErr<<std::endl;

    outFile.close();
	#endif

    hipFree(x);
    if(!inplace)
    {
        hipFree(y);
    }
    hipFree(fbuffer);
    hipFree(bbuffer);

    // Destroy plans
    rocfft_plan_destroy(forward);
    rocfft_plan_destroy(backward);

    rocfft_cleanup();
}
