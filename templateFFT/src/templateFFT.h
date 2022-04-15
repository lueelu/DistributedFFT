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
	uint64_t* bufferSize;//array of buffers sizes in bytes
	uint64_t* inputBufferSize;//array of input buffers sizes in bytes, if isInputFormatted is enabled
	uint64_t* outputBufferSize;//array of output buffers sizes in bytes, if isOutputFormatted is enabled
	uint64_t* kernelSize;//array of kernel buffers sizes in bytes, if performConvolution is enabled
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
	uint64_t aimThreads;//aim at this many threads per block. Default 128
	uint64_t numSharedBanks;//how many banks shared memory has. Default 32
	uint64_t inverseReturnToInputBuffer;//return data to the input buffer in inverse transform (0 - off, 1 - on). isInputFormatted must be enabled
	uint64_t numberBatches;// N - used to perform multiple batches of initial data. Default 1
	uint64_t useUint64;//use 64-bit addressing mode in generated kernels
	uint64_t registerBoost;
	uint64_t doublePrecision; 
	uint64_t useLUT; 
	uint64_t reorderFourStep;
    uint64_t makeForwardPlanOnly;
	uint64_t makeInversePlanOnly; 
	uint64_t maxComputeWorkGroupCount[3];
	uint64_t maxComputeWorkGroupSize[3]; 
	uint64_t maxThreadsNum; 
	uint64_t sharedMemorySizeStatic; //available for  allocation shared memory size, in bytes
	uint64_t sharedMemorySize; //available for allocation shared memory size, in bytes
	uint64_t sharedMemorySizePow2; //power of 2 which is less or equal to sharedMemorySize, in bytes
	uint64_t warpSize; //number of threads per warp/wavefront.
	uint64_t halfThreads;//Intel fix
	uint64_t allocateTempBuffer; //buffer allocated by app automatically if needed to reorder Four step algorithm. Parameter to check if it has been allocated
	int64_t maxCodeLength; //specify how big can be buffer used for code generation (in char). Default 1000000 chars.
	int64_t maxTempLength; //specify how big can be buffer used for intermediate string sprintfs be (in char). Default 5000 chars. If code segfaults for some reason - try increasing this number.
} FFTConfiguration;


typedef struct {
	uint64_t size[3];
	uint64_t localSize[3];
	uint64_t fftDim;
	uint64_t inverse;
	uint64_t actualInverse;
	uint64_t zeropad[2];
	uint64_t axis_id;
	uint64_t axis_upload_id;
	uint64_t registers_per_thread;
	uint64_t registers_per_thread_per_radix[14];
	uint64_t min_registers_per_thread;
	uint64_t readToRegisters;
	uint64_t writeFromRegisters;
	uint64_t LUT;
	uint64_t performR2C;
	uint64_t performR2CmultiUpload;
	uint64_t performDCT;
	uint64_t frequencyZeropadding;
	uint64_t performZeropaddingFull[3]; 
	uint64_t performZeropaddingInput[3]; 
	uint64_t performZeropaddingOutput[3]; 
	uint64_t fft_zeropad_left_full[3];
	uint64_t fft_zeropad_left_read[3];
	uint64_t fft_zeropad_left_write[3];
	uint64_t fft_zeropad_right_full[3];
	uint64_t fft_zeropad_right_read[3];
	uint64_t fft_zeropad_right_write[3];
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
	uint64_t numCoordinates;
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
	uint64_t symmetricKernel;
	uint64_t supportAxis;
	uint64_t cacheShuffle;
	uint64_t registerBoost;
	uint64_t warpSize;
	uint64_t numSharedBanks;
	uint64_t resolveBankConflictFirstStages;
	uint64_t sharedStrideBankConflictFirstStages;
	uint64_t sharedStrideReadWriteConflict;
	uint64_t maxSharedStride;
	uint64_t axisSwapped;
	uint64_t mergeSequencesR2C;
	uint64_t numBuffersBound[4];
	uint64_t performBufferSetUpdate;
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
} FFTSpecializationConstantsLayout;

typedef struct {
	uint32_t coordinate;
	uint32_t batch;
	uint32_t workGroupShift[3];
} FFTPushConstantsLayoutUint32;
typedef struct {
	uint64_t coordinate;
	uint64_t batch;
	uint64_t workGroupShift[3];
} FFTPushConstantsLayoutUint64;


typedef struct {
	uint64_t numBindings;
	uint64_t axisBlock[4];
	uint64_t groupedBatch;
	FFTSpecializationConstantsLayout specializationConstants;
	FFTPushConstantsLayoutUint32 pushConstantsUint32;
	FFTPushConstantsLayoutUint64 pushConstants;
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

 FFTResult FFTCheckUpdateBufferSet(FFTApplication* app, FFTAxis* axis, uint64_t planStage, FFTLaunchParams* launchParams);
 FFTResult FFTUpdateBufferSet(FFTApplication* app, FFTPlan* FFTPlan, FFTAxis* axis, uint64_t axis_id, uint64_t axis_upload_id, uint64_t inverse);
 FFTResult AppendLine(FFTSpecializationConstantsLayout* sc);
 FFTResult MulComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2, const char* temp);
 FFTResult SubComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2);
 FFTResult AddComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2);
FFTResult AddComplexInv(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2);
FFTResult FMAComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num, const char* in_2);
FFTResult MulComplexNumber(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num);
FFTResult MovComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in);
FFTResult ShuffleComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2, const char* temp);
FFTResult ShuffleComplexInv(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2, const char* temp); 
FFTResult DivComplexNumber(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num);
  FFTResult AddReal(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2);
  FFTResult MovReal(FFTSpecializationConstantsLayout* sc, const char* out, const char* in);
  FFTResult ModReal(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num);
  FFTResult SubReal(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2);
  FFTResult MulReal(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2);
  FFTResult SharedStore(FFTSpecializationConstantsLayout* sc, const char* id, const char* in);
  FFTResult SharedLoad(FFTSpecializationConstantsLayout* sc, const char* out, const char* id);
 FFTResult inlineRadixKernelFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t radix, uint64_t stageSize, double stageAngle, char** regID);
  FFTResult appendExtensions(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeInputMemory, const char* floatTypeOutputMemory, const char* floatTypeKernelMemory);
  FFTResult appendPushConstant(FFTSpecializationConstantsLayout* sc, const char* type, const char* name);
  FFTResult appendConstant(FFTSpecializationConstantsLayout* sc, const char* type, const char* name, const char* defaultVal, const char* LFending);
  FFTResult AppendLineFromInput(FFTSpecializationConstantsLayout* sc, const char* in);
  FFTResult appendConstantsFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType); 
  FFTResult appendSinCos20(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType);
  FFTResult appendConversion(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeDifferent);
  FFTResult appendPushConstantsFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType);
  FFTResult appendInputLayoutFFT(FFTSpecializationConstantsLayout* sc, uint64_t id, const char* floatTypeMemory, uint64_t inputType);
  FFTResult appendOutputLayoutFFT(FFTSpecializationConstantsLayout* sc, uint64_t id, const char* floatTypeMemory, uint64_t outputType);
  FFTResult appendLUTLayoutFFT(FFTSpecializationConstantsLayout* sc, uint64_t id, const char* floatType);
  FFTResult appendSharedMemoryFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t sharedType);
  FFTResult appendInitialization(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t initType);
  FFTResult appendZeropadStart(FFTSpecializationConstantsLayout* sc);
  FFTResult appendZeropadEnd(FFTSpecializationConstantsLayout* sc);
  FFTResult appendBarrierFFT(FFTSpecializationConstantsLayout* sc, uint64_t numTab);
  FFTResult appendBoostThreadDataReorder(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t shuffleType, uint64_t start);
  FFTResult appendRadixStageNonStrided(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix);
  FFTResult appendRadixStageStrided(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix);
  FFTResult appendRadixStage(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t shuffleType);
  FFTResult appendRegisterBoostShuffle(FFTSpecializationConstantsLayout* sc, const char* floatType, uint64_t stageSize, uint64_t stageRadixPrev, uint64_t stageRadix, double stageAngle);
  FFTResult appendRadixShuffleNonStrided(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext);
  FFTResult appendRadixShuffleStrided(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext);
  FFTResult appendRadixShuffle(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext, uint64_t shuffleType);
  FFTResult appendReorder4StepWrite(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t reorderType);
  FFTResult indexInputFFT(FFTSpecializationConstantsLayout* sc, const char* uintType, uint64_t inputType, const char* index_x, const char* index_y, const char* coordinate, const char* batchID);
  FFTResult indexOutputFFT(FFTSpecializationConstantsLayout* sc, const char* uintType, uint64_t outputType, const char* index_x, const char* index_y, const char* coordinate, const char* batchID);
  FFTResult appendZeropadStartReadWriteStage(FFTSpecializationConstantsLayout* sc, uint64_t readStage);
  FFTResult appendZeropadEndReadWriteStage(FFTSpecializationConstantsLayout* sc);
  FFTResult appendReorder4StepRead(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t reorderType);
 FFTResult appendReadDataFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint64_t readType);
 FFTResult appendWriteDataFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint64_t writeType);
 FFTResult FFTScheduler(FFTApplication* app, FFTPlan* FFTPlan, uint64_t axis_id, uint64_t supportAxis);
 void deleteAxis(FFTApplication* app, FFTAxis* axis);
 void deleteFFT(FFTApplication* app);
 void freeShaderGenFFT(FFTSpecializationConstantsLayout* sc);
 FFTResult shaderGenFFT(char* output, FFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeInputMemory, const char* floatTypeOutputMemory, const char* floatTypeKernelMemory, const char* uintType, uint64_t type);
 FFTResult FFTPlanAxis(FFTApplication* app, FFTPlan* FFTPlan, uint64_t axis_id, uint64_t axis_upload_id, uint64_t inverse);
 FFTResult initializeFFT(FFTApplication* app, FFTConfiguration inputLaunchConfiguration);
 FFTResult dispatchEnhanced(FFTApplication* app, FFTAxis* axis, uint64_t* dispatchBlock, FFTLaunchArgs* launchArgs);
 FFTResult FFTAppend(FFTApplication* app, int inverse, FFTLaunchParams* launchParams);
FFTResult setFFTArgs(GPU* GPU, FFTApplication* app, FFTLaunchParams* launchParams, int inverse);
hipError_t launchFFTKernel(FFTApplication* app, int inverse);
