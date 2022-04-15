# A Highly Efficient GPU Framework for Fast Fourier Transform across Mixed Nodes

## About
This library is being developed to support highly efficient distributed FFT computation on multicore CPU and GPU architectures. The evaluation section in the manuscript is all included in this library.

## Structure
- templateFFT  (templated-based FFT library on one single GPU)
- 3dmpifft_roc (optimozed distributed FFT; backened: rocFFT)
- 3dmpifft_opt (optimized distributed FFT; backened: template-based FFT)
- heffte       (current state-of-art distributed FFT library heffte)


## Build requirements:
- Compiler (GCC and hipcc)
- CMake (>v3.0)
- ROCm platform (>v4.2)
- rocfft (>v4.2)
- OpenMPI (>v4.0.0) and UCX (>1.11.0)

## Getting Start Guide
1. Compile and run batched ffts on one single GPU:
```
cd template/batchTest && make 
bash runTest1D_opt.sh    # performance test of batched 1D templated-based FFT
bash runTest1D_roc.sh    # performance test of batched 1D rocfft
```
Evaluation results in the manuscript are recorded in csv folder and sample kernels is presented in kernel folder.

2. Typical CMake build for distributed FFT follow the steps:
```
cd 3dmpifft_opt (or 3dmpifft_roc) && mkdir build && cd build
cmake -DMPI_DIR=$OMPI_DIR -DCMAKE_BUILD_TYPE=RELEASE ..
make -j
```
Run script (customize <MPI_DIR> in line 2 of this script and nodelist for multinodes) and sample output:
```
sh speedTest.sh <MPI-RANK> <X> <Y> <Z> 

t0: 0.003835, t1: 0.003627, t2: 0.016155, t3: 0.005500, total: 0.029118
t0: 0.004212, t1: 0.004101, t2: 0.014490, t3: 0.005405, total: 0.028208
t0: 0.004113, t1: 0.003239, t2: 0.014504, t3: 0.005496, total: 0.027352
t0: 0.004251, t1: 0.003561, t2: 0.014023, t3: 0.005489, total: 0.027324
----------------------------------------------------------------------------- 
distributed FFT performance test
----------------------------------------------------------------------------- 
Size:             512x512x512
MPI ranks:        4
Forward FFT time: 0.0281308 (s)
Performance:      644.112 GFlops/s
Max error:        4.21468e-15

```
t0, t1, t2 and t3 are batched 2D FFT on dimension YZ, local transpose, all-to-all communication and batched 1D FFT on dimension X, respectively. Time of all-to-all communication depends mainly on the MPI library and the PCI, which means that t2 may fluctuate. Despite of our optimization on t0 and t3, the overall GFlops may also vary massively because of mere 0.01s in all-to-all communication. Therefore, data in the manuscript is just the reference value of repeated experiments.

3. Comparison with heffte (no need to compile the whole project):
```
cd heffte/heffteBenchmark/benchmarks && make
sh heffteSpeed.sh <MPI-RANK> <X> <Y> <Z> 

----------------------------------------------------------------------------- 
heFFTe performance test
----------------------------------------------------------------------------- 
Backend:   rocfft
Size:      512x512x512
MPI ranks:    4
Grids: (1, 2, 2)  (2, 1, 2)  (2, 2, 1)  (1, 2, 2)  
Time per run: 0.0558535 (s)
Performance:  324.41 GFlops/s
Memory usage: 2560MB/rank
Tolerance:    1e-11
Max error:    4.88555e-15
```


