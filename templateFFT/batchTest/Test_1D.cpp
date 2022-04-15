#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>

#include "templateFFT.h"

int num_iter;
int N;
int X;
int Y;
int Z;
int printResult;

FFTResult sample_FFT_1D_double(GPU* GPU, uint64_t file_output, FILE* output)
{
	FFTResult resFFT = FFT_SUCCESS;


	hipError_t res = hipSuccess;

	if (file_output)
		fprintf(output, "1 - FFT + iFFT C2C 1D in double precision LUT\n");
	printf("1 - FFT + iFFT C2C 1D in double precision LUT\n");


	uint64_t input_size = N;
	double* buffer_input = (double*)malloc(sizeof(double) * 2 * N);
	if (!buffer_input) return FFT_ERROR_MALLOC_FAILED;

	double* buffer_output = (double*)malloc(sizeof(double) * 2 * N);
	if (!buffer_output) return FFT_ERROR_MALLOC_FAILED;

	// init the data
	for (uint64_t i = 0; i < input_size; i++) {
        buffer_input[2 * i] = ((int) i) + 1.0;
        buffer_input[2 * i + 1] = 0.0;
    }

        hipSetDevice(3);
	// double* buffer_input = (double*)malloc((uint64_t)8 * 2 * (uint64_t)pow(2, 27));
	// if (!buffer_input) return FFT_ERROR_MALLOC_FAILED;
	// for (uint64_t i = 0; i < 2 * (uint64_t)pow(2, 27); i++) {
	// 	buffer_input[i] = (double)2 * ((double)rand()) / RAND_MAX - 1.0;
	// }
	// double* buffer_output = (double*)malloc((uint64_t)8 * 2 * (uint64_t)pow(2, 27));

    /*   Set the configuration   */
	FFTConfiguration configuration = {};
	FFTApplication app = {};
	configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
	configuration.size[0] = X; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
	// configuration.size[1] = Y;
    configuration.size[1] = Y;
	configuration.size[2] = Z;
	configuration.doublePrecision = true;
	configuration.useLUT = true; //use twiddle factor table
	configuration.makeForwardPlanOnly = false;  
	configuration.device = &GPU->device;

		
	uint64_t bufferSize = (uint64_t)sizeof(double) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2];

	hipDoubleComplex* buffer = 0;
	res = hipMalloc((void **)&buffer, bufferSize);
	if (res != hipSuccess) return FFT_ERROR_FAILED_TO_ALLOCATE;
	configuration.bufferSize = &bufferSize; //Allocate buffer and copy data to GPU
	res = hipMemcpy(buffer, buffer_input, bufferSize, hipMemcpyHostToDevice);
	configuration.buffer = (void **)&buffer;

	if (res != hipSuccess) return FFT_ERROR_FAILED_TO_COPY;


	/*   Initialize, get FFT Kernel   */
	resFFT = initializeFFT(&app, configuration);
	if (resFFT != FFT_SUCCESS) return resFFT;
	FFTLaunchParams launchParams = {};
	resFFT = setFFTArgs(GPU, &app, &launchParams, 0); //Set the launch kernel args(grid dim...) 0 is forward FFT and 1 is inverse
	if (resFFT != FFT_SUCCESS) return resFFT;

	//Warm up...Run FFT
	res = launchFFTKernel(&app, 0);
	// Copy Data from GPU to host
	res = hipMemcpy(buffer_output, buffer, bufferSize, hipMemcpyDeviceToHost);
    if (res != hipSuccess) return FFT_ERROR_FAILED_TO_COPY;

	if (printResult == 1) {
        for (int i = 0; i < 8; i++) {
        std::cout << "element " << i << " input:  (" << buffer_input[2 * i] << ","
                  << buffer_input[2 * i + 1] << ")"
                  << " output: (" << buffer_output[2 * i] << "," << buffer_output[2 * i + 1] << ")"
                  << std::endl;
        }
        for (uint64_t i = input_size - 8; i < input_size; i++) {
            std::cout << "element " << i << " input:  (" << buffer_input[2 * i] << ","
                      << buffer_input[2 * i + 1] << ")" << " output: ("
                      << buffer_output[2 * i] << "," << buffer_output[2 * i + 1] << ")" << std::endl;
        }
    }


    /*   Performance Test  */

	// Another buffer for next FFT, if the next FFT configuration is the same, just change the buffer and launch the kernel.
	hipDoubleComplex* tmpbuffer = 0;
    res = hipMalloc((void **)&tmpbuffer, bufferSize);
	res = hipMemcpy(tmpbuffer, buffer_input, bufferSize, hipMemcpyHostToDevice);

	hipEvent_t start, stop;
	float elapsedTime = 0.f;
	double averageTime = 0.0;

	hipEventCreate(&start);
	hipEventCreate(&stop);
	hipEventRecord(start, NULL);
	for(int i = 0; i< num_iter; i++){
		//change the data
		app.configuration.buffer = (void**)&tmpbuffer;
		res = launchFFTKernel(&app, 0);
	}
	hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsedTime, start, stop);

	double opscount = Y * (double)5 * X * log(X) / log(2.0);

	printf("FFT: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " Buffer: %f MB avg_hip_time: %0.6f ms Gflops: %0.6f num_iter: %d \n", configuration.size[0], configuration.size[1], configuration.size[2], bufferSize / 1024.0 / 1024.0, elapsedTime/num_iter, opscount/(1e6 * elapsedTime / num_iter), num_iter);


	/*   Inverse FFT*/
	#if 1
	res = hipMemcpy(buffer, buffer_output, bufferSize, hipMemcpyHostToDevice);
	resFFT = setFFTArgs(GPU, &app, &launchParams, 1); 
	if (resFFT != FFT_SUCCESS) return resFFT;
	app.configuration.buffer = (void **)&buffer;
	res = launchFFTKernel(&app, 1);

	res = hipMemcpy(buffer_output, buffer, bufferSize, hipMemcpyDeviceToHost);
    if (res != hipSuccess) return FFT_ERROR_FAILED_TO_COPY;

	if (printResult == 1) {
        for (int i = 0; i < 8; i++) {
        std::cout << "element " << i << " input:  (" << buffer_input[2 * i] << ","
                  << buffer_input[2 * i + 1] << ")"
                  << " output: (" << buffer_output[2 * i] << "," << buffer_output[2 * i + 1] << ")"
                  << std::endl;
        }
        for (uint64_t i = input_size - 8; i < input_size; i++) {
            std::cout << "element " << i << " input:  (" << buffer_input[2 * i] << ","
                      << buffer_input[2 * i + 1] << ")" << " output: ("
                      << buffer_output[2 * i] << "," << buffer_output[2 * i + 1] << ")" << std::endl;
        }
    }

	double tmp1, tmp2, tmp3 ,maxErr = 0.0;
	for (int i = 0; i < N; i++) {
		tmp1 = buffer_input[2*i]-buffer_output[2*i]/X;
		tmp2 = buffer_input[2*i+1]-buffer_output[2*i+1]/X;
		tmp3 = sqrt( tmp1*tmp1 + tmp2*tmp2 );
		maxErr = maxErr >= tmp3 ? maxErr : tmp3;
	}
	std::cout<<"Max error: "<< maxErr<<std::endl;

	#endif

	#if 0
	uint64_t num_tot_transfers = 0;
	for (uint64_t i = 0; i < configuration.FFTdim; i++)
		num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
	num_tot_transfers *= 2;
	// printf("num_tot_transfers = %ld\n", num_tot_transfers);
	char filename[30] = "batch_result1D.csv";
	std::ofstream outFile;
	outFile.open(filename, std::ios::app);
	outFile<<configuration.size[0]<<','<<configuration.size[1]<<','<<configuration.size[2]<<','<<bufferSize / 1024.0 / 1024.0<<','<<elapsedTime/num_iter<<','<<opscount/(1e6 * elapsedTime / num_iter)<<','<<num_iter<<','<<bufferSize / 1024.0 / 1024.0 / 1.024 * num_tot_transfers / (elapsedTime/num_iter)<<','<<maxErr<<std::endl;
    outFile.close();
	#endif

	hipFree(buffer);
	deleteFFT(&app);
	free(buffer_input);
	free(buffer_output);
	return resFFT;
}



int main(int argc, char *argv[]) {
    GPU GPU = {};
    X = atoi(argv[1]);
    Y = atoi(argv[2]);
    Z = atoi(argv[3]);
    num_iter = atoi(argv[4]);
    printResult = atoi(argv[5]);

	Y = (uint64_t)(64 * 32 * (uint64_t)pow(2, 15))/ X;

    N = X * Y * Z;

    bool file_output = false;
    FILE *output = NULL;
    int sscanf_res = 0;

    FFTResult resFFT = sample_FFT_1D_double(&GPU, file_output, output);

    if (resFFT != FFT_SUCCESS) printf("Error! id= %d\n", resFFT);


    return 0;
}
