#include "fft_mpi_3d_api.h"

void fft_mpi_init(const longInt64 *N, int iniDeviceNumInNode, MPI_Comm comm, int &newDeviceCount, int &newDeviceCountInNode, longInt64 dataCountInNode[]) 
{
    int mpi_size, mpi_rank;

    MPI_CHECK(MPI_Comm_size(comm, &mpi_size));
    MPI_CHECK(MPI_Comm_rank(comm, &mpi_rank));
    // get the proper device num within node
    getProperDeviceNum(N, iniDeviceNumInNode, mpi_size, mpi_rank, newDeviceCount, newDeviceCountInNode);

    // get data count for each device
    getDataCountForNode(dataCountInNode, N, mpi_rank, mpi_size, newDeviceCount, newDeviceCountInNode);

    // enable gpu direct
    for (int i = 0; i < newDeviceCountInNode; ++i) {
        ROCM_CHECK(hipSetDevice(i));
        for (int j = 0; j < newDeviceCountInNode; ++j) {
            if (i != j ) {
                int canAccessPeer = 0;
                ROCM_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, i, j));
                if (canAccessPeer) {
                    ROCM_CHECK(hipDeviceEnablePeerAccess(j, 0));
                }
            }
        }
    }

#ifdef ENABLE_RCCL
    // init rccl
    ncclComm_t rccl_comms[newDeviceCountInNode];
    ncclUniqueId unId;
    if (mpi_rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&unId));
    }
    MPI_Bcast(&unId, sizeof(unId), MPI_BYTE, 0, comm);
#endif

}

fft_mpi_3d_plan_p fft_mpi_plan_dft_c2c_3d(longInt64 n0, longInt64 n1, longInt64 n2, Complex *in, Complex *out, Complex **node_data, MPI_Comm comm, int devIdx, int devCountInNode, int totalDevCount, int direction) 
{
    fft_mpi_3d_plan_p plan = new fft_mpi_3d_plan;
    plan->N[0] = n0;
    plan->N[1] = n1;
    plan->N[2] = n2;

    plan->direction = direction;

    plan->locGPUIdx = devIdx;
    plan->devCountInNode = devCountInNode;
    plan->totalDevCount = totalDevCount;

    MPI_CHECK(MPI_Comm_rank(comm, &plan->mpiRank));
    MPI_CHECK(MPI_Comm_size(comm, &plan->mpiSize));

    plan->globalDevIdx = plan->mpiRank * ceil((double)totalDevCount/plan->mpiSize) + devIdx;

    if (plan->globalDevIdx == totalDevCount - 1)
        plan->isLastDevice = 1;
    else
        plan->isLastDevice = 0;

    plan->maxDataCountInDevice = getMaxDataCount(n0, n1, n2, totalDevCount, plan->isLastDevice);

    plan->inDev = in;
    plan->outDev = out;
    if (plan->outDev == NULL || plan->outDev == plan->inDev) {
        plan->isInplace = 1;
        plan->bufferDev2 = plan->inDev;
    }
    else {
        plan->isInplace = 0;
        plan->bufferDev2 = plan->outDev;
    }
    ROCM_CHECK(hipMalloc(&plan->bufferDev1, plan->maxDataCountInDevice * sizeof(Complex)));
    plan->nodeDataDev = node_data;
    plan->nodeDataDev[plan->locGPUIdx] = plan->bufferDev2;
    

    plan->tInfo.rcount = new longInt64[totalDevCount];
    plan->tInfo.roffset = new longInt64[totalDevCount];
    plan->tInfo.scount = new longInt64[totalDevCount];
    plan->tInfo.soffset = new longInt64[totalDevCount];

    plan->lastExchangeN2 = n2;
    plan->lastExchangeN0 = n0 - (totalDevCount - 1) * ceil((double)n0 / totalDevCount);
    plan->lastExchangeN1 = n1 - (totalDevCount - 1) * ceil((double)n1 / totalDevCount);

    if (!plan->isLastDevice) {
        int normalDataCount = ceil((double)n0 / totalDevCount) * ceil((double)n1 / totalDevCount) * n2;
        for (int i = 0; i < totalDevCount - 1; ++i) {
            plan->tInfo.rcount[i] = normalDataCount;
            plan->tInfo.scount[i] = normalDataCount;
            plan->tInfo.roffset[i] = i * normalDataCount;
            plan->tInfo.soffset[i] = i * normalDataCount;
        }
        // for the last device
        if (plan->direction == FORWARD) {
            plan->tInfo.rcount[totalDevCount - 1] = plan->lastExchangeN0 * ceil((double)n1 / totalDevCount) * n2;
            plan->tInfo.scount[totalDevCount - 1] = ceil((double)n0 / totalDevCount) * plan->lastExchangeN1 * n2;
        }
        else if (plan->direction == BACKWARD) {
            plan->tInfo.rcount[totalDevCount - 1] = ceil((double)n0 / totalDevCount) * plan->lastExchangeN1 * n2;
            plan->tInfo.scount[totalDevCount - 1] = plan->lastExchangeN0 * ceil((double)n1 / totalDevCount) * n2;
        }
        plan->tInfo.roffset[totalDevCount - 1] = (totalDevCount - 1) * normalDataCount;
        plan->tInfo.soffset[totalDevCount - 1] = (totalDevCount - 1) * normalDataCount;
    }
    else {
        int normal_recv_count, normal_send_count;
        if (plan->direction == FORWARD) {
            normal_recv_count = ceil((double)n0 / totalDevCount) * plan->lastExchangeN1 * n2;
            normal_send_count = plan->lastExchangeN0 * ceil((double)n1 / totalDevCount) * n2;
        }
        else if (plan->direction == BACKWARD) {
            normal_recv_count = plan->lastExchangeN0 * ceil((double)n1 / totalDevCount) * n2;
            normal_send_count = ceil((double)n0 / totalDevCount) * plan->lastExchangeN1 * n2;
        }
        for (int i = 0; i < totalDevCount - 1; ++i) {
            plan->tInfo.rcount[i] = normal_recv_count;
            plan->tInfo.scount[i] = normal_send_count;
            plan->tInfo.roffset[i] = i * normal_recv_count;
            plan->tInfo.soffset[i] = i * normal_send_count;
        }
        plan->tInfo.rcount[totalDevCount - 1] = plan->lastExchangeN0 * plan->lastExchangeN1 * n2;
        plan->tInfo.scount[totalDevCount - 1] = plan->lastExchangeN0 * plan->lastExchangeN1 * n2;
        plan->tInfo.roffset[totalDevCount - 1] = (totalDevCount - 1) * normal_recv_count;
        plan->tInfo.soffset[totalDevCount - 1] = (totalDevCount - 1) * normal_send_count;
    }

    setFFTPlans(plan);

    ROCM_CHECK(hipStreamCreate(&plan->stream1));
    ROCM_CHECK(hipStreamCreate(&plan->stream2));

    return plan;
}

void fft_mpi_destroy_plan(fft_mpi_3d_plan_p plan) 
{
    delete[] plan->tInfo.rcount;
    delete[] plan->tInfo.scount;
    delete[] plan->tInfo.roffset;
    delete[] plan->tInfo.soffset;

    ROCM_CHECK(hipStreamDestroy(plan->stream1));
    ROCM_CHECK(hipStreamDestroy(plan->stream2));

    if (plan->hipPlanX != NULL)
        HIPFFT_CHECK(hipfftDestroy(plan->hipPlanX));
    if (plan->hipPlanY != NULL)
        HIPFFT_CHECK(hipfftDestroy(plan->hipPlanY));
    if (plan->hipPlanZ != NULL)
        HIPFFT_CHECK(hipfftDestroy(plan->hipPlanZ));
    if (plan->hipPlanYZ != NULL)
        HIPFFT_CHECK(hipfftDestroy(plan->hipPlanYZ));

    if (plan->workBufSizeX) {
        hipFree(plan->workBufX);
        rocfft_execution_info_destroy(plan->rocPlanInfoX);
    }
    if (plan->workBufSizeYZ) {
        hipFree(plan->workBufYZ);
        rocfft_execution_info_destroy(plan->rocPlanInfoYZ);
    }
    if (plan->rocPlanX != NULL)
        ROCFFT_CHECK(rocfft_plan_destroy(plan->rocPlanX));
    if (plan->rocPlanYZ != NULL)
        ROCFFT_CHECK(rocfft_plan_destroy(plan->rocPlanYZ));

    ROCM_CHECK(hipFree(plan->bufferDev1));

    delete plan;
}

void fft_mpi_execute_dft_3d_c2c(fft_mpi_3d_plan_p p) 
{
    if (p->direction == FORWARD) {
        double t0 = -MPI_Wtime();
        fftZY(p);
        t0 += MPI_Wtime();
        double t1 = -MPI_Wtime();
        localTransposeUneven(p);
        t1 += MPI_Wtime();
#pragma omp barrier
        // MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        double t2 = -MPI_Wtime();
        slabAlltoall(p);
        t2 += MPI_Wtime();
#pragma omp barrier
        // MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        double t3 = -MPI_Wtime();
        fftX(p);
        t3 += MPI_Wtime();
        printf("t0: %lf, t1: %lf, t2: %lf, t3: %lf, total: %lf\n", t0, t1, t2, t3, t0+t1+t2+t3);
    }
    else if (p->direction == BACKWARD) {
        fftX(p);
        // debugLocalData(p, 1, 3);
#pragma omp barrier
        // MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        slabAlltoall(p);
#pragma omp barrier
        // MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        localTransposeUneven(p);
        fftZY(p);
        int x_remain = p->isLastDevice? p->lastExchangeN0: ceil((double)p->N[0] / p->totalDevCount);
        call_scale_element((double*)p->bufferDev2, p->N[0], p->N[1], p->N[2], x_remain);
        ROCM_CHECK(hipDeviceSynchronize());
    }
}

Complex* fft_mpi_alloc_local_memory(int count, int flag) 
{
    Complex *data;
    if (flag == ALLOC_CPU) {
        data = (Complex*)malloc(count * sizeof(Complex));
    }
    else if (flag == ALLOC_DEV) {
        ROCM_CHECK(hipMalloc(&data, count * sizeof(Complex)));
    }
    else {
        printf("Fail to allocate memory!\n");
        exit(EXIT_FAILURE);
    }
    return data;
}

void getProperDeviceNum(const longInt64 *N, int iniDeviceNumInNode, int mpiSize, int mpiRank, int &newDeviceCount, int &newDeviceCountInNode) {

    int real_device_count;
    ROCM_CHECK(hipGetDeviceCount(&real_device_count));
    if (iniDeviceNumInNode > real_device_count) {
        printf("The number of GPUs in rank %d is less than %d, so it will be set to %d (equal to your real device count).\n", mpiRank, iniDeviceNumInNode, real_device_count);
        iniDeviceNumInNode = real_device_count;
    }

    int newDeviceNum = iniDeviceNumInNode * mpiSize, newDeviceNumInNode = iniDeviceNumInNode;

    // get proper device num
    if (N[0] % (iniDeviceNumInNode * mpiSize) != 0) {
        int dataCountInDevice = N[0] / (iniDeviceNumInNode * mpiSize) + 1;
        newDeviceNum = N[0] / dataCountInDevice;
        if (N[0] % dataCountInDevice != 0) {
            newDeviceNum += 1;
        }
        // allocate device to each process
        newDeviceNumInNode = newDeviceNum / mpiSize;
        // could not allocate evenly
        if (newDeviceNum % mpiSize != 0) {  
            int remainDevice = newDeviceNum % mpiSize;
            if (mpiRank < remainDevice) {
                newDeviceNumInNode += 1;
            }
        }
    }
    
    // one situation may exist: one device per node, and must reduce one process, we will solve it in the future

    newDeviceCount = newDeviceNum;
    newDeviceCountInNode = newDeviceNumInNode;

    if (newDeviceCountInNode == 0) {
        printf("could not support this distribution of data, exit!!\n");
        exit(EXIT_FAILURE);
    }
    printf("allocate %d devices to node %d\n", newDeviceCountInNode, mpiRank);    
        
}

void getDataCountForNode(longInt64 dataCountInNode[], const longInt64 N[], int mpiRank, int mpiSize, int deviceCount, int deviceCountInNode) 
{
    longInt64 normal_data_count = ceil((double)N[0] / deviceCount) * N[1] * N[2];

    for (int i = 0; i < deviceCountInNode; ++i) {
        if (mpiRank == mpiSize - 1 && i == deviceCountInNode - 1) {
            dataCountInNode[i] = N[0] * N[1] * N[2] - normal_data_count * (deviceCount - 1);
        }
        else {
            dataCountInNode[i] = normal_data_count;
        }
        printf("data count in device %d of node %d: %lld\n", i, mpiRank, dataCountInNode[i]);
    }
}

longInt64 getMaxDataCount(int n0, int n1, int n2, int totalDevCount, bool isLastDevice) 
{
    int n0Dev, n1Dev;
    if (isLastDevice) {
        n0Dev = n0 - (totalDevCount - 1) * ceil((double)n0 / totalDevCount);
        n1Dev = n1 - (totalDevCount - 1) * ceil((double)n1 / totalDevCount);
    }
    else {
        n0Dev = ceil((double)n0 / totalDevCount);
        n1Dev = ceil((double)n1/ totalDevCount);
    }

    // N0 direction
    longInt64 max_data_count = -1;
    max_data_count = n0Dev * n1 * n2;

    // N1 direction
    longInt64 tmp_data_count = -1;
    tmp_data_count = n0 * n1Dev * n2;

    if (max_data_count <= tmp_data_count) {
        return tmp_data_count;
    }
    else {
        return max_data_count;
    }

}

void setFFTPlans(fft_mpi_3d_plan_p plan) 
{
    // forward fft of z dimension
    int z_inembed[1] = { plan->N[2] }, z_onembed[1] = { plan->N[2]}, 
        z_batch = (plan->isLastDevice? plan->lastExchangeN0 : ceil((double)plan->N[0] / plan->totalDevCount)) * plan->N[1];
    HIPFFT_CHECK(hipfftPlanMany(&plan->hipPlanZ, 1, &plan->N[2], z_inembed, 1, plan->N[2], z_onembed, 1, plan->N[2], HIPFFT_Z2Z, z_batch));

    // forword fft of y dimension
    int y_inembed[1] = { plan->N[1] }, y_onembed[1] = { plan->N[1] }, y_stride = plan->N[2], y_dis = 1,
        y_batch = plan->N[2];
    HIPFFT_CHECK(hipfftPlanMany(&plan->hipPlanY, 1, &plan->N[1], y_inembed, y_stride, y_dis, y_onembed, y_stride, y_dis, HIPFFT_Z2Z, y_batch));

    // 2D FFT
    HIPFFT_CHECK(hipfftPlan2d(&plan->hipPlanYZ, plan->N[1], plan->N[2], HIPFFT_Z2Z));

    // forward fft of x dimension
    int x_inembed[1] = { plan->N[0] }, x_onembed[1] = { plan->N[0] }, 
        x_stride = plan->isLastDevice? plan->lastExchangeN1 * plan->N[2]: ceil((double)plan->N[1] / plan->totalDevCount) * plan->N[2],
        x_dis = 1, x_batch = x_stride;
    // printf("node_%d_gpu_%d\tstride: %d, batch: %d\n", plan->mpiRank, plan->locGPUIdx, x_stride, x_batch);
    HIPFFT_CHECK(hipfftPlanMany(&plan->hipPlanX, 1, &plan->N[0], x_inembed, x_stride, x_dis, x_onembed, x_stride, x_dis, HIPFFT_Z2Z, x_batch));

    // set for rocfft
    // 2d
    int tmp = ceil((double)plan->N[0] / plan->totalDevCount),
        times = plan->isLastDevice? (plan->N[0] - (plan->totalDevCount - 1) * tmp): tmp;
    size_t roclengthYZ[2] = {(size_t)plan->N[2], (size_t)plan->N[1]}, numberOfTranYZ = times;
    if (plan->direction == FORWARD)
        ROCFFT_CHECK(rocfft_plan_create(&plan->rocPlanYZ, rocfft_placement_notinplace, rocfft_transform_type_complex_forward, rocfft_precision_double, 2, roclengthYZ, numberOfTranYZ, NULL));
    else if (plan->direction == BACKWARD)
        ROCFFT_CHECK(rocfft_plan_create(&plan->rocPlanYZ, rocfft_placement_notinplace, rocfft_transform_type_complex_inverse, rocfft_precision_double, 2, roclengthYZ, numberOfTranYZ, NULL));
    // Check if the plan requires a work buffer
    rocfft_plan_get_work_buffer_size(plan->rocPlanYZ, &plan->workBufSizeYZ);
    if (plan->workBufSizeYZ)
    {
        rocfft_execution_info_create(&plan->rocPlanInfoYZ);
        hipMalloc(&plan->workBufYZ, plan->workBufSizeYZ);
        rocfft_execution_info_set_work_buffer(plan->rocPlanInfoYZ, plan->workBufYZ, plan->workBufSizeYZ);
    }
    // 1d
    // rocfft_plan_description desX;
    // const size_t inOffsetX[1] = {0}, outOffsetX[1] = {0}, 
    //         inStrideX[1] = {plan->isLastDevice? (size_t)plan->lastExchangeN1 * plan->N[2]: (size_t)ceil((double)plan->N[1] / plan->totalDevCount) * plan->N[2]},
    //         outStrideX[1] = {plan->isLastDevice? (size_t)plan->lastExchangeN1 * plan->N[2]: (size_t)ceil((double)plan->N[1] / plan->totalDevCount) * plan->N[2]};
    // ROCFFT_CHECK(rocfft_plan_description_create(&desX));
    // ROCFFT_CHECK(rocfft_plan_description_set_data_layout(desX, rocfft_array_type_complex_interleaved, rocfft_array_type_complex_interleaved, inOffsetX, outOffsetX, 1, inStrideX, 1, 1, outStrideX, 1));
    // size_t roclengthX[1] = {(size_t)plan->N[0]}, numberOfTranX = inStrideX[0];
    // if (plan->direction == FORWARD)
    //     ROCFFT_CHECK(rocfft_plan_create(&plan->rocPlanX, rocfft_placement_inplace, rocfft_transform_type_complex_forward, rocfft_precision_double, 1, roclengthX, numberOfTranX, desX));
    // else if (plan->direction == BACKWARD)
    //     ROCFFT_CHECK(rocfft_plan_create(&plan->rocPlanX, rocfft_placement_notinplace, rocfft_transform_type_complex_inverse, rocfft_precision_double, 1, roclengthX, numberOfTranX, desX));

    size_t timesX = plan->isLastDevice? plan->lastExchangeN1 * plan->N[2]: (ceil((double)plan->N[1] / plan->totalDevCount) * plan->N[2]);
    const size_t lengthX[1] = {(size_t)plan->N[0]};
    if (plan->direction == FORWARD)
        ROCFFT_CHECK(rocfft_plan_create(&plan->rocPlanX, rocfft_placement_notinplace, rocfft_transform_type_complex_forward, rocfft_precision_double, 1, lengthX, timesX, NULL));
    else if (plan->direction == BACKWARD)
        ROCFFT_CHECK(rocfft_plan_create(&plan->rocPlanX, rocfft_placement_notinplace, rocfft_transform_type_complex_inverse, rocfft_precision_double, 1, lengthX, timesX, NULL));

    // Check if the plan requires a work buffer
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(plan->rocPlanX, &plan->workBufSizeX));
    if (plan->workBufSizeX)
    {
        ROCFFT_CHECK(rocfft_execution_info_create(&plan->rocPlanInfoX));
        ROCM_CHECK(hipMalloc(&plan->workBufX, plan->workBufSizeX));
        ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(plan->rocPlanInfoX, plan->workBufX, plan->workBufSizeX));
    }
}

void outputPlanInfo(fft_mpi_3d_plan_p plan) 
{
    std::ofstream of;
    char filename[50];
    sprintf(filename, "rank_%d_gpu_%d.txt", plan->mpiRank, plan->locGPUIdx);
    of.open(filename);

    char content[2048];
    sprintf(content, "N0: %d, N1: %d, N2: %d\n \
            globalDeviceIndex: %d, localGPUIndex: %d, deviceCountInNode: %d, totalDeviceCount: %d\n \
            mpiRank: %d, mpiSize: %d\n \
            isLastDevice: %d, direction: %d\n \
            maxDataCountInDevice: %lld\n \
            lastExchangeN0: %lld, lastExchangeN1: %lld, lastExchangeN2: %lld\n",
            plan->N[0], plan->N[1], plan->N[2], 
            plan->globalDevIdx, plan->locGPUIdx, plan->devCountInNode, plan->totalDevCount, 
            plan->mpiRank, plan->mpiSize, 
            plan->isLastDevice, plan->direction, 
            plan->maxDataCountInDevice, 
            plan->lastExchangeN0, plan->lastExchangeN1, plan->lastExchangeN2);
    of << content << std::endl << std::endl;

    // output the global transpose info
    for (int i = 0; i < plan->totalDevCount; ++i) {
        char tmp[100];
        sprintf(tmp, "send to %d, count: %lld, offset: %lld, receice from %d, count: %lld, offset: %lld\n", 
                i, plan->tInfo.scount[i], plan->tInfo.soffset[i], i, plan->tInfo.rcount[i], plan->tInfo.roffset[i]);
        of << tmp;
    }

    of.close();
}

void fftZY(fft_mpi_3d_plan_p plan) 
{ 
    // // z dimension
    // HIPFFT_CHECK(hipfftExecZ2Z(plan->hipPlanZ, (hipfftDoubleComplex*)plan->inDev, (hipfftDoubleComplex*)plan->bufferDev1, HIPFFT_FORWARD));
    
    // // y dimension
    // int y_batch = plan->isLastDevice? plan->lastExchangeN0: ceil((double)plan->N[0] / plan->totalDevCount);
    // for (int i = 0; i < y_batch; ++i) {
    //     HIPFFT_CHECK(hipfftExecZ2Z(plan->hipPlanY, (hipfftDoubleComplex*)plan->bufferDev1, (hipfftDoubleComplex*)plan->bufferDev2, HIPFFT_FORWARD));
    // }

    // int tmp = ceil((double)plan->N[0] / plan->totalDevCount),
    //     times = plan->isLastDevice? (plan->N[0] - (plan->totalDevCount - 1) * tmp): tmp;
    // for (int i = 0; i < times; ++i) {
    //     longInt64 offset = i * plan->N[1] * plan->N[2];
    //     HIPFFT_CHECK(hipfftExecZ2Z(plan->hipPlanYZ, (hipfftDoubleComplex *)plan->inDev + offset, (hipfftDoubleComplex *)plan->bufferDev2 + offset, HIPFFT_FORWARD));
    // }
    if (plan->direction == FORWARD)
        ROCFFT_CHECK(rocfft_execute(plan->rocPlanYZ, (void**)&plan->inDev, (void**)&plan->bufferDev2, plan->rocPlanInfoYZ));
        // ROCM_CHECK(hipMemcpy(plan->bufferDev2, plan->inDev, plan->maxDataCountInDevice * sizeof(Complex), hipMemcpyDeviceToDevice));
    else if (plan->direction == BACKWARD)
        ROCFFT_CHECK(rocfft_execute(plan->rocPlanYZ, (void**)&plan->bufferDev1, (void**)&plan->bufferDev2, plan->rocPlanInfoYZ));
    ROCM_CHECK(hipDeviceSynchronize());
}

void fftX(fft_mpi_3d_plan_p plan) 
{

    // x dimension
    // HIPFFT_CHECK(hipfftExecZ2Z(plan->hipPlanX, (hipfftDoubleComplex*)plan->bufferDev2, (hipfftDoubleComplex*)plan->bufferDev2, HIPFFT_FORWARD));
    int y_size = plan->isLastDevice? plan->lastExchangeN1: ceil((double)plan->N[1] / plan->totalDevCount);
    if (plan->direction == FORWARD) {
        // call_local_transpose_z_to_x_strided_forward((double*)plan->bufferDev2, (double*)plan->bufferDev1, plan->N[0], y_size, plan->N[2]);
        // const int tSize[3] = { plan->N[2], y_size, plan->N[0] },
        //             permutation[3] = {2, 0, 1};
        // int result = cut_transpose3d((data_t*)plan->bufferDev1, (data_t*)plan->bufferDev2, tSize, permutation, 1);
        // if (result < 0) {
        //     printf("fail to do stride transpose in x dimension!\n");
        //     exit(EXIT_FAILURE);
        // }
        // debugLocalData(plan, 1, 0);
        call_local_transpose_z_to_x_strided_forward_opt((double*)plan->bufferDev2, (double*)plan->bufferDev1, plan->N[0], y_size, plan->N[2]);
        ROCFFT_CHECK(rocfft_execute(plan->rocPlanX, (void**)&plan->bufferDev1, (void**)&plan->bufferDev2, plan->rocPlanInfoX));
        // ROCM_CHECK(hipMemcpy(plan->bufferDev2, plan->bufferDev1, plan->maxDataCountInDevice * sizeof(Complex), hipMemcpyDeviceToDevice));
    }
    else if (plan->direction == BACKWARD) {
        // ROCM_CHECK(hipMemcpy(plan->bufferDev2, plan->inDev, plan->maxDataCountInDevice * sizeof(Complex), hipMemcpyDeviceToDevice));
        ROCFFT_CHECK(rocfft_execute(plan->rocPlanX, (void**)&plan->inDev, (void**)&plan->bufferDev2, plan->rocPlanInfoX));
        // call_local_transpose_z_to_x_strided_backward((double*)plan->bufferDev2, (double*)plan->bufferDev1, plan->N[0], y_size, plan->N[2]);
        call_local_transpose_z_to_x_strided_backward_opt((double*)plan->bufferDev2, (double*)plan->bufferDev1, plan->N[0], y_size, plan->N[2]);
    }

    ROCM_CHECK(hipDeviceSynchronize());
}

void localTransposeUneven(fft_mpi_3d_plan_p plan) 
{
    int x_size, y_size, z_size, y_divide, y_last;
    if (!plan->isLastDevice) {
        x_size = ceil((double)plan->N[0] / plan->totalDevCount);    
    }
    else {
        x_size = plan->N[0] - (plan->totalDevCount - 1) * ceil((double)plan->N[0] / plan->totalDevCount);
    }
    y_size = plan->N[1];
    z_size = plan->N[2];
    y_divide = ceil((double)plan->N[1] / plan->totalDevCount);    
    y_last = plan->lastExchangeN1;

    if (plan->direction == FORWARD)
        // call_slab_local_transpose_z_to_x_uneven((double*)plan->bufferDev2, (double*)plan->bufferDev1, x_size, y_size, z_size, y_divide, y_last, plan->totalDevCount, plan->direction);
        call_slab_local_transpose_z_to_x_uneven_optimized((double*)plan->bufferDev2, (double*)plan->bufferDev1, x_size, y_size, z_size, y_divide, y_last, plan->totalDevCount, plan->direction);
    else if (plan->direction == BACKWARD)
        // call_slab_local_transpose_z_to_x_uneven((double*)plan->bufferDev2, (double*)plan->bufferDev1, x_size, y_size, z_size, y_divide, y_last, plan->totalDevCount, plan->direction);
        call_slab_local_transpose_z_to_x_uneven_optimized((double*)plan->bufferDev2, (double*)plan->bufferDev1, x_size, y_size, z_size, y_divide, y_last, plan->totalDevCount, plan->direction);


    ROCM_CHECK(hipDeviceSynchronize());
    // debugLocalData(plan, 1, 1);
}

void slabAlltoall(fft_mpi_3d_plan_p plan) 
{
    // from plan->locGPUIdx to i
    for (int i = 0; i < plan->devCountInNode; ++i) {
        int recv_idx = plan->mpiRank * ceil((double)plan->totalDevCount / plan->mpiSize) + i;
        longInt64 send_offset = plan->tInfo.soffset[recv_idx],
            send_bytes = plan->tInfo.scount[recv_idx] * sizeof(Complex),
            recv_offset;
        if (recv_idx == plan->totalDevCount - 1) {
            if (plan->direction == FORWARD)
                recv_offset = plan->globalDevIdx * ceil((double)plan->N[0] / plan->totalDevCount) * plan->lastExchangeN1 * plan->N[2];
            else if (plan->direction == BACKWARD)
                recv_offset = plan->globalDevIdx * plan->lastExchangeN0 * ceil((double)plan->N[1] / plan->totalDevCount) * plan->N[2];
        }
        else
            recv_offset = plan->globalDevIdx * ceil((double)plan->N[0] / plan->totalDevCount) * ceil((double)plan->N[1] / plan->totalDevCount) * plan->N[2];
        // printf("local rank: %d, from %d to %d, so: %lld, sc: %lld, ro: %lld, rc: %lld\n", plan->mpiRank, plan->locGPUIdx, i, send_offset, send_bytes/16, recv_offset, send_bytes/16);
        // if (i % 2)
            // ROCM_CHECK(hipMemcpyPeerAsync(plan->nodeDataDev[i]+recv_offset, i, plan->bufferDev1+send_offset, plan->locGPUIdx, send_bytes, plan->stream1));
            ROCM_CHECK(hipMemcpyAsync(plan->nodeDataDev[i]+recv_offset, plan->bufferDev1+send_offset, send_bytes, hipMemcpyDeviceToDevice, plan->stream1));
            // ROCM_CHECK(hipMemcpy(plan->nodeDataDev[i]+recv_offset, plan->bufferDev1+send_offset, send_bytes, hipMemcpyDeviceToDevice));

        // else
            // ROCM_CHECK(hipMemcpyPeerAsync(plan->nodeDataDev[i]+recv_offset, i, plan->bufferDev1+send_offset, plan->locGPUIdx, send_bytes, plan->stream2));
            // ROCM_CHECK(hipMemcpyAsync(plan->nodeDataDev[i]+recv_offset, plan->bufferDev1+send_offset, send_bytes, hipMemcpyDeviceToDevice, plan->stream2));
    }
    // to other mpi processes
    // ROCM_CHECK(hipDeviceSynchronize());
    // MPI_Request recv_reqs[plan->totalDevCount], send_reqs[plan->totalDevCount];

    MPI_Request *recv_reqs = new MPI_Request[plan->totalDevCount - plan->devCountInNode], *send_reqs = new MPI_Request[plan->totalDevCount - plan->devCountInNode];
    int req_cnt = 0;
    for (int i = 0; i < plan->mpiSize; ++i) {
        int dev_count,
            normal_dev_count = (ceil)((double)plan->totalDevCount / plan->mpiSize);
        if (i != plan->mpiSize - 1) {
            dev_count = normal_dev_count;
        }
        else {
            dev_count = plan->totalDevCount - (plan->mpiSize-1) * normal_dev_count;
        }
        if (i != plan->mpiRank) {
            for (int j = 0; j < dev_count; ++j) {
                int dev_idx = i * normal_dev_count + j;
                longInt64 roffset = plan->tInfo.roffset[dev_idx], soffset = plan->tInfo.soffset[dev_idx],
                    rcount = plan->tInfo.rcount[dev_idx], scount = plan->tInfo.scount[dev_idx];
                int send_tag = plan->globalDevIdx * 10 + dev_idx, recv_tag = dev_idx * 10 + plan->globalDevIdx;
#pragma omp critical 
{
                // MPI_CHECK(MPI_Irecv(plan->bufferDev2+roffset, rcount*2, MPI_DOUBLE, i, recv_tag, MPI_COMM_WORLD, &recv_reqs[req_cnt]));
                MPI_CHECK(MPI_Irecv(plan->bufferDev2+roffset, rcount*2, MPI_DOUBLE, i, recv_tag, MPI_COMM_WORLD, &recv_reqs[req_cnt]));
                MPI_CHECK(MPI_Isend(plan->bufferDev1+soffset, scount*2, MPI_DOUBLE, i, send_tag, MPI_COMM_WORLD, &send_reqs[req_cnt++]));
}
                // MPI_CHECK(MPI_Recv(plan->bufferDev2+roffset, rcount*2, MPI_DOUBLE, i, recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                // MPI_CHECK(MPI_Send(plan->bufferDev1+soffset, scount*2, MPI_DOUBLE, i, send_tag, MPI_COMM_WORLD));
                // printf("rank: %d\trecv: from %d, tag: %d\tsend: to %d, tag: %d\tsc: %lld, so: %lld, rc: %lld, ro: %lld\n", plan->mpiRank, i, recv_tag, i, send_tag, scount, soffset, rcount, roffset);
            }
        }
    }
    // ROCM_CHECK(hipStreamSynchronize(plan->stream1));
    // ROCM_CHECK(hipStreamSynchronize(plan->stream2));
    ROCM_CHECK(hipDeviceSynchronize());
    // MPI_CHECK(MPI_Waitall(req_cnt, send_reqs, MPI_STATUSES_IGNORE));
#pragma omp critical 
{
    MPI_CHECK(MPI_Waitall(req_cnt, recv_reqs, MPI_STATUSES_IGNORE));
    MPI_CHECK(MPI_Waitall(req_cnt, send_reqs, MPI_STATUSES_IGNORE));
}
//     int tmpCount = 0, flag[req_cnt];
//     memset(flag, 0, sizeof(int) * req_cnt);
//     while (tmpCount < req_cnt) {
//         for (int i = 0; i < req_cnt; ++i) {
//             if (flag[i] == 1)
//                 continue;
// #pragma omp critical 
// {
//             MPI_CHECK(MPI_Test(&recv_reqs[i], &flag[i], MPI_STATUS_IGNORE));
// }
//             if (flag[i] == 1)
//                 ++tmpCount;
//         }
//     }
}

void debugLocalData(fft_mpi_3d_plan_p plan, int flag, int type)
{
    Complex *data_ptr;
    if (flag == 1) {
        data_ptr = plan->bufferDev1;
    }
    else if (flag == 2) {
        data_ptr = plan->bufferDev2;
    }
    else {
        printf("Fail to debug local data, exit!\n");
    }

    // int bytes_size = this->data_count_per_gpu * sizeof(Complex);
    // int data_count = (plan->isLastDevice? this->last_exchange_n0: ceil((double)plan->N[0] / plan->totalDevCount)) * plan->N[1] * plan->N[2],
    int data_count = plan->maxDataCountInDevice,
        bytes_size = data_count * sizeof(Complex);
    Complex *tmp_data = (Complex*)malloc(bytes_size);
    ROCM_CHECK(hipMemcpy(tmp_data, data_ptr, bytes_size, hipMemcpyDeviceToHost));
    // printf("GPU: %d ", plan->locGPUIdx);

    std::ofstream of;
    char test[100];
    sprintf(test, "node_%d_gpu_%d.csv", plan->mpiRank, plan->locGPUIdx);
    of.open(test);

    for (int i = 0; i < data_count; ++i) {
        if (type == 0) {
            int z = (int)tmp_data[i][0] % plan->N[2], y = (int)tmp_data[i][0] / plan->N[2] % plan->N[1], x = (int)tmp_data[i][0] / plan->N[2] / plan->N[1] % plan->N[0];
            // printf("%d: (%d %d %d) ", i, x, y, z);
            // printf("(%d, %d, %d)/", x, y, z);
            sprintf(test, "(%d, %d, %d)/", x, y, z);
        }
        else if (type == 1) {
            sprintf(test, "[%lf, %lf]/", tmp_data[i][0], tmp_data[i][1]);
        }
        else if (type == 2) {
            sprintf(test, "[%lf, %lf]/", tmp_data[i][0] / (plan->N[0] * plan->N[1] * plan->N[2]), tmp_data[i][1] / (plan->N[0] * plan->N[1] * plan->N[2]));
        }
        else if (type == 3) {
            sprintf(test, "[%lf, %lf]/", tmp_data[i][0] / plan->N[0], tmp_data[i][1] / plan->N[0]);
        }
        // sprintf(test, "[%lf, %lf]/", tmp_data[i][0], tmp_data[i][1]);
        of << test;
    }
    of << "\n";

    of.close();
}

