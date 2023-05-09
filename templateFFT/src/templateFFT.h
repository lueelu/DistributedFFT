#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <vector>
#include <hip/hip_runtime.h>


typedef enum FFTResult {
	FFT_SUCCESS = 0,
	FFT_ERROR_MALLOC_FAILED = 1,
	FFT_ERROR_INSUFFICIENT_CODE_BUFFER = 2,
	FFT_ERROR_INSUFFICIENT_TEMP_BUFFER = 3,
	FFT_ERROR_PLAN_NOT_INITIALIZED = 4,
	FFT_ERROR_NULL_TEMP_PASSED = 5,
	FFT_ERROR_INVALID_PHYSICAL_DEVICE = 1001,
	FFT_ERROR_INVALID_DEVICE = 1002,
	FFT_ERROR_INVALID_FENCE = 1005,
	FFT_ERROR_ONLY_FORWARD_FFT_INITIALIZED = 1006,
	FFT_ERROR_ONLY_INVERSE_FFT_INITIALIZED = 1007,
	FFT_ERROR_INVALID_CONTEXT = 1008,
	FFT_ERROR_INVALID_PLATFORM = 1009,
	FFT_ERROR_EMPTY_FFTdim = 2001,
	FFT_ERROR_EMPTY_size = 2002,
	FFT_ERROR_EMPTY_bufferSize = 2003,
	FFT_ERROR_EMPTY_buffer = 2004,
	FFT_ERROR_EMPTY_tempBufferSize = 2005,
	FFT_ERROR_EMPTY_tempBuffer = 2006,
	FFT_ERROR_EMPTY_inputBufferSize = 2007,
	FFT_ERROR_EMPTY_inputBuffer = 2008,
	FFT_ERROR_EMPTY_outputBufferSize = 2009,
	FFT_ERROR_EMPTY_outputBuffer = 2010,
	FFT_ERROR_EMPTY_kernelSize = 2011,
	FFT_ERROR_EMPTY_kernel = 2012,
	FFT_ERROR_UNSUPPORTED_RADIX = 3001,
	FFT_ERROR_UNSUPPORTED_FFT_LENGTH = 3002,
	FFT_ERROR_FAILED_TO_ALLOCATE = 4001,
	FFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE = 4020,
	FFT_ERROR_FAILED_TO_CREATE_DEVICE = 4021,
	FFT_ERROR_FAILED_TO_CREATE_BUFFER = 4024,
	FFT_ERROR_FAILED_TO_ALLOCATE_MEMORY = 4025,
	FFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY = 4026,
	FFT_ERROR_FAILED_TO_FIND_MEMORY = 4027,
	FFT_ERROR_FAILED_TO_SYNCHRONIZE = 4028,
	FFT_ERROR_FAILED_TO_COPY = 4029,
	FFT_ERROR_FAILED_TO_CREATE_PROGRAM = 4030,
	FFT_ERROR_FAILED_TO_COMPILE_PROGRAM = 4031,
	FFT_ERROR_FAILED_TO_GET_CODE_SIZE = 4032,
	FFT_ERROR_FAILED_TO_GET_CODE = 4033,
	FFT_ERROR_FAILED_TO_DESTROY_PROGRAM = 4034,
	FFT_ERROR_FAILED_TO_LOAD_MODULE = 4035,
	FFT_ERROR_FAILED_TO_GET_FUNCTION = 4036,
	FFT_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY = 4037,
	FFT_ERROR_FAILED_TO_MODULE_GET_GLOBAL = 4038,
	FFT_ERROR_FAILED_TO_LAUNCH_KERNEL = 4039,
	FFT_ERROR_FAILED_TO_EVENT_RECORD = 4040,
	FFT_ERROR_FAILED_TO_ADD_NAME_EXPRESSION = 4041,
	FFT_ERROR_FAILED_TO_INITIALIZE = 4042,
	FFT_ERROR_FAILED_TO_SET_DEVICE_ID = 4043,
	FFT_ERROR_FAILED_TO_GET_DEVICE = 4044,
	FFT_ERROR_FAILED_TO_CREATE_CONTEXT = 4045,
	FFT_ERROR_FAILED_TO_SET_KERNEL_ARG = 4047,
	FFT_ERROR_FAILED_TO_ENUMERATE_DEVICES = 4050,
	FFT_ERROR_FAILED_TO_GET_ATTRIBUTE = 4051
} FFTResult;

typedef struct {
	hipDevice_t device;
	hipCtx_t context;
	uint64_t device_id;
} GPU;

typedef struct {
	void** buffer;
	void** tempBuffer;
	void** inputBuffer;
	void** outputBuffer;
	void** kernel;
} FFTLaunchParams;


typedef struct {
	uint64_t FFTdim; 
	uint64_t size[3]; 

	hipDevice_t* device;
	uint64_t* bufferSize;
	uint64_t* inputBufferSize;
	uint64_t* outputBufferSize;
	uint64_t* kernelSize;
	uint64_t coalescedMemory;
	uint64_t bufferStride[3];
	uint64_t inputBufferStride[3];
	uint64_t outputBufferStride[3];

	void** buffer;
	void** tempBuffer;
	void** inputBuffer;
	void** outputBuffer;
	void** kernel;

	void *originInput;
	uint64_t* tempBufferSize;
    uint64_t userTempBuffer;
	uint64_t normalize;

	uint64_t disableReorderFourStep;
	uint64_t aimThreads;
	uint64_t numSharedBanks;
	uint64_t inverseReturnToInputBuffer;
	uint64_t numberBatches;
	uint64_t useUint64;
	uint64_t regAd;
	uint64_t doublePrecision; 
	uint64_t useLUT; 
	uint64_t reorderFourStep;
    uint64_t makeForwardPlanOnly;
	uint64_t makeInversePlanOnly; 
	uint64_t maxComputeWorkGroupCount[3];
	uint64_t maxComputeWorkGroupSize[3]; 
	uint64_t maxThreadsNum; 
	uint64_t sharedMemorySizeStatic; 
	uint64_t sharedMemorySize;
	uint64_t sharedMemorySizePow2; 
	uint64_t warpSize; 
	uint64_t halfThreads;
	uint64_t allocateTempBuffer; 
	int64_t maxCodeLength; 
	int64_t maxTempLength; 
} FFTConfiguration;


typedef struct {
	uint64_t size[3];
	uint64_t localSize[3];
	uint64_t dim;
	uint64_t inverse;
	uint64_t actualInverse;
	uint64_t axis_id;
	uint64_t axis_upload_id;
	uint64_t threadRegister;
	uint64_t threadRadixRegister[14];
	uint64_t threadRegisterMin;
	uint64_t readToRegisters;
	uint64_t writeFromRegisters;
	uint64_t LUT;
	uint64_t inputStride[5];
	uint64_t outputStride[5];
	uint64_t fft_dim_full;
	uint64_t stageStartSize;
	uint64_t firstStageStartSize;
	uint64_t fft_dim_x;
	uint64_t numStages;
	uint64_t stageRadix[20];
	uint64_t inputOffset;
	uint64_t outputOffset;
	uint64_t reorderFourStep;
	uint64_t performWorkGroupShift[3];
	uint64_t inputBufferBlockNum;
	uint64_t inputBufferBlockSize;
	uint64_t outputBufferBlockNum;
	uint64_t outputBufferBlockSize;
	uint64_t kernelBlockNum;
	uint64_t kernelBlockSize;
	uint64_t matrixConvolution; 
	uint64_t numBatches;
	uint64_t numKernels;
	uint64_t usedSharedMemory;
	uint64_t sharedMemSize;
	uint64_t sharedMemSizePow2;
	uint64_t normalize;
	uint64_t complexSize;
	uint64_t maxStageSumLUT;
	uint64_t unroll;
	uint64_t convolutionStep;
	uint64_t supportAxis;
	uint64_t regAd;
	uint64_t warpSize;
	uint64_t numSharedBanks;
	uint64_t conflictStages;
	uint64_t conflictStride;
	uint64_t conflictShared;
	uint64_t maxSharedStride;
	uint64_t axisSwapped;
	uint64_t mergeSequencesR2C;
	uint64_t numBuffersBound[4];
	uint64_t bufferUpdate;
	uint64_t useUint64;
	char** regIDs;
	char* disableThreadsStart;
	char* disableThreadsEnd;
	char sdataID[50];
	char inoutID[50];
	char combinedID[50];
	char gl_LocalInvocationID_x[50];
	char gl_LocalInvocationID_y[50];
	char gl_LocalInvocationID_z[50];
	char gl_GlobalInvocationID_x[200];
	char gl_GlobalInvocationID_y[200];
	char gl_GlobalInvocationID_z[200];
	char tshuffle[50];
	char sharedStride[50];
	char gl_WorkGroupSize_x[50];
	char gl_WorkGroupSize_y[50];
	char gl_WorkGroupSize_z[50];
	char gl_WorkGroupID_x[50];
	char gl_WorkGroupID_y[50];
	char gl_WorkGroupID_z[50];
	char tempReg[50];
	char stageInvocationID[50];
	char blockInvocationID[50];
	char temp[50];
	char w[50];
	char iw[50];
	char locID[13][40];
	char* output;
	char* tempStr;
	int64_t tempLen;
	int64_t currentLen;
	int64_t maxCodeLength;
	int64_t maxTempLength;
} FFTLayout;

typedef struct {
	uint32_t coordinate;
	uint32_t batch;
	uint32_t workGroupShift[3];
} FFTLayoutUint32;
typedef struct {
	uint64_t coordinate;
	uint64_t batch;
	uint64_t workGroupShift[3];
} FFTLayoutUint64;


typedef struct {
	uint64_t numBindings;
	uint64_t axisBlock[4];
	uint64_t groupedBatch;
	FFTLayout layout;
	FFTLayoutUint32 layoutUnit32;
	FFTLayoutUint64 layoutUnit64;
	uint64_t updatePushConstants;

	void** inputBuffer;
	void** outputBuffer;

	void** tmpInputBuffer;
	void** tmpoutPutBuffer;
	hipModule_t FFTModule;
	hipFunction_t FFTKernel;
	void* bufferLUT;
	hipDeviceptr_t consts_addr;

	uint64_t bufferLUTSize;
	uint64_t referenceLUT;

} FFTAxis;

typedef struct
{
	void *args[3];
	unsigned int gridSize[3];
    unsigned int blockSize[3];
	unsigned int sharedMem;

}FFTLaunchArgs;

typedef struct {
	uint64_t actualFFTSizePerAxis[3][3];
	uint64_t numAxisUploads[3];
	uint64_t axisSplit[3][4];
	FFTAxis axes[3][4];
	FFTLaunchArgs launchArgs[3][4];
} FFTPlan;

typedef struct {
	FFTConfiguration configuration;
	FFTPlan* localFFTPlan;
	FFTPlan* localFFTPlan_inverse; //additional inverse plan
} FFTApplication;

 FFTResult FFTCheckBuffer(FFTApplication* app, FFTAxis* axis, uint64_t planStage, FFTLaunchParams* launchParams);
 FFTResult FFTUpdateBuffer(FFTApplication* app, FFTPlan* FFTPlan, FFTAxis* axis, uint64_t axis_id, uint64_t axis_upload_id, uint64_t inverse);
 FFTResult AppendLine(FFTLayout* lt);
 FFTResult MulComplex(FFTLayout* lt, const char* out, const char* in_1, const char* in_2, const char* temp);
 FFTResult SubComplex(FFTLayout* lt, const char* out, const char* in_1, const char* in_2);
 FFTResult AddComplex(FFTLayout* lt, const char* out, const char* in_1, const char* in_2);
FFTResult AddComplexInv(FFTLayout* lt, const char* out, const char* in_1, const char* in_2);
FFTResult FMAComplex(FFTLayout* lt, const char* out, const char* in_1, const char* in_num, const char* in_2);
FFTResult MulComplexNumber(FFTLayout* lt, const char* out, const char* in_1, const char* in_num);
FFTResult MovComplex(FFTLayout* lt, const char* out, const char* in);
FFTResult ShuffleComplex(FFTLayout* lt, const char* out, const char* in_1, const char* in_2, const char* temp);
FFTResult ShuffleComplexInv(FFTLayout* lt, const char* out, const char* in_1, const char* in_2, const char* temp); 
FFTResult DivComplexNumber(FFTLayout* lt, const char* out, const char* in_1, const char* in_num);
  FFTResult AddReal(FFTLayout* lt, const char* out, const char* in_1, const char* in_2);
  FFTResult MovReal(FFTLayout* lt, const char* out, const char* in);
  FFTResult ModReal(FFTLayout* lt, const char* out, const char* in_1, const char* in_num);
  FFTResult SubReal(FFTLayout* lt, const char* out, const char* in_1, const char* in_2);
  FFTResult MulReal(FFTLayout* lt, const char* out, const char* in_1, const char* in_2);
  FFTResult SharedStore(FFTLayout* lt, const char* id, const char* in);
  FFTResult SharedLoad(FFTLayout* lt, const char* out, const char* id);
 FFTResult inlineRadixKernelFFT(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t radix, uint64_t stageSize, double stageAngle, char** regID);
  FFTResult appendExtensions(FFTLayout* lt, const char* floatType, const char* floatTypeInputMemory, const char* floatTypeOutputMemory, const char* floatTypeKernelMemory);
  FFTResult appendPushConstant(FFTLayout* lt, const char* type, const char* name);
  FFTResult appendConstant(FFTLayout* lt, const char* type, const char* name, const char* defaultVal, const char* LFending);
  FFTResult AppendLineFromInput(FFTLayout* lt, const char* in);
  FFTResult appendConstantsFFT(FFTLayout* lt, const char* floatType, const char* uintType); 
  FFTResult appendSinCos20(FFTLayout* lt, const char* floatType, const char* uintType);
  FFTResult appendConversion(FFTLayout* lt, const char* floatType, const char* floatTypeDifferent);
  FFTResult appendPushConstantsFFT(FFTLayout* lt, const char* floatType, const char* uintType);
  FFTResult appendInputLayoutFFT(FFTLayout* lt, uint64_t id, const char* floatTypeMemory, uint64_t inputType);
  FFTResult appendOutputLayoutFFT(FFTLayout* lt, uint64_t id, const char* floatTypeMemory, uint64_t outputType);
  FFTResult appendLUTLayoutFFT(FFTLayout* lt, uint64_t id, const char* floatType);
  FFTResult appendSharedMemoryFFT(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t sharedType);
  FFTResult appendInitialization(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t initType);
  FFTResult appendBarrier(FFTLayout* lt, uint64_t numTab);
  FFTResult threadDataOrder(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t shuffleType, uint64_t start);
  FFTResult radixNonStrided(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix);
  FFTResult radixStrided(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix);
  FFTResult appendRadixStage(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t shuffleType);
  FFTResult appendregAdShuffle(FFTLayout* lt, const char* floatType, uint64_t stageSize, uint64_t stageRadixPrev, uint64_t stageRadix, double stageAngle);
  FFTResult appendRadixShuffleNonStrided(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext);
  FFTResult appendRadixShuffleStrided(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext);
  FFTResult appendRadixShuffle(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext, uint64_t shuffleType);
  FFTResult appendReorder4StepWrite(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t reorderType);
  FFTResult indexInputFFT(FFTLayout* lt, const char* uintType, uint64_t inputType, const char* index_x, const char* index_y, const char* coordinate, const char* batchID);
  FFTResult indexOutputFFT(FFTLayout* lt, const char* uintType, uint64_t outputType, const char* index_x, const char* index_y, const char* coordinate, const char* batchID);
  FFTResult appendReorder4StepRead(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t reorderType);
 FFTResult appendReadDataFFT(FFTLayout* lt, const char* floatType, const char* floatTypeMemory, const char* uintType, uint64_t readType);
 FFTResult appendWriteDataFFT(FFTLayout* lt, const char* floatType, const char* floatTypeMemory, const char* uintType, uint64_t writeType);
 FFTResult FFTScheduler(FFTApplication* app, FFTPlan* FFTPlan, uint64_t axis_id, uint64_t supportAxis);
 void deleteAxis(FFTApplication* app, FFTAxis* axis);
 void deleteFFT(FFTApplication* app);
 void freeShaderGenFFT(FFTLayout* lt);
 FFTResult shaderGenFFT(char* output, FFTLayout* lt, const char* floatType, const char* floatTypeInputMemory, const char* floatTypeOutputMemory, const char* floatTypeKernelMemory, const char* uintType, uint64_t type);
 FFTResult FFTPlanAxis(FFTApplication* app, FFTPlan* FFTPlan, uint64_t axis_id, uint64_t axis_upload_id, uint64_t inverse);
 FFTResult initializeFFT(FFTApplication* app, FFTConfiguration inputLaunchConfiguration);
 FFTResult dispatchEnhanced(FFTApplication* app, FFTAxis* axis, uint64_t* dispatchBlock, FFTLaunchArgs* launchArgs);
 FFTResult FFTAppend(FFTApplication* app, int inverse, FFTLaunchParams* launchParams);
FFTResult setFFTArgs(GPU* GPU, FFTApplication* app, FFTLaunchParams* launchParams, int inverse);
hipError_t launchFFTKernel(FFTApplication* app, int inverse);
