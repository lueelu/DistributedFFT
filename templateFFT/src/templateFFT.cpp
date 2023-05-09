#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>

#include "templateFFT.h"



FFTResult FFTCheckBuffer(FFTApplication* app, FFTAxis* axis, uint64_t planStage, FFTLaunchParams* launchParams) {
	uint64_t performUpdate = planStage;
	if (!planStage) {
		if (launchParams != 0) {
			if ((launchParams->buffer != 0) && (app->configuration.buffer != launchParams->buffer)) {
				app->configuration.buffer = launchParams->buffer;
				performUpdate = 1;
			}
			if ((launchParams->inputBuffer != 0) && (app->configuration.inputBuffer != launchParams->inputBuffer)) {
				app->configuration.inputBuffer = launchParams->inputBuffer;
				performUpdate = 1;
			}
			if ((launchParams->outputBuffer != 0) && (app->configuration.outputBuffer != launchParams->outputBuffer)) {
				app->configuration.outputBuffer = launchParams->outputBuffer;
				performUpdate = 1;
			}
			if ((launchParams->tempBuffer != 0) && (app->configuration.tempBuffer != launchParams->tempBuffer)) {
				app->configuration.tempBuffer = launchParams->tempBuffer;
				performUpdate = 1;
			}
			if ((launchParams->kernel != 0) && (app->configuration.kernel != launchParams->kernel)) {
				app->configuration.kernel = launchParams->kernel;
				performUpdate = 1;
			}
			if (app->configuration.inputBuffer == 0) app->configuration.inputBuffer = app->configuration.buffer;
			if (app->configuration.outputBuffer == 0) app->configuration.outputBuffer = app->configuration.buffer;
		}
	}
	if (planStage) {
		if (app->configuration.buffer == 0) performUpdate = 0;
	}
	else {
		if (app->configuration.buffer == 0) return FFT_ERROR_EMPTY_buffer;
	}
	if (performUpdate) {
		if (planStage) axis->layout.bufferUpdate = 1;
		else {
			if (!app->configuration.makeInversePlanOnly) {
				for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
					for (uint64_t j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++)
						app->localFFTPlan->axes[i][j].layout.bufferUpdate = 1;
				}
			}
			if (!app->configuration.makeForwardPlanOnly) {
				for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
					for (uint64_t j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++)
						app->localFFTPlan_inverse->axes[i][j].layout.bufferUpdate = 1;
				}
			}
		}
	}
	return FFT_SUCCESS;
}





FFTResult FFTUpdateBuffer(FFTApplication* app, FFTPlan* FFTPlan, FFTAxis* axis, uint64_t axis_id, uint64_t axis_upload_id, uint64_t inverse) {
	if (axis->layout.bufferUpdate) {
		uint64_t storageComplexSize;
		storageComplexSize = (2 * sizeof(double));

		for (uint64_t i = 0; i < axis->numBindings; ++i) {
			for (uint64_t j = 0; j < axis->layout.numBuffersBound[i]; ++j) {
				if (i == 0) {
					uint64_t bufferId = 0;
					uint64_t offset = j;
					if ((FFTPlan->numAxisUploads[axis_id] > 1))
						if (axis_upload_id > 0) axis->inputBuffer = app->configuration.buffer;
						else axis->inputBuffer = app->configuration.tempBuffer;
					else axis->inputBuffer = app->configuration.buffer;
				}
				if (i == 1) {
					if ((FFTPlan->numAxisUploads[axis_id] > 1)) {
						if (axis_upload_id == 1) axis->outputBuffer = app->configuration.tempBuffer;
						else axis->outputBuffer = app->configuration.buffer;
					}
					else axis->outputBuffer = app->configuration.buffer;
				}
			}
		}
		axis->layout.bufferUpdate = 0;
	}
	return FFT_SUCCESS;
}

FFTResult AppendLine(FFTLayout* lt) {
	if (lt->tempLen < 0) return FFT_ERROR_INSUFFICIENT_TEMP_BUFFER;
	if (lt->currentLen + lt->tempLen > lt->maxCodeLength) return FFT_ERROR_INSUFFICIENT_CODE_BUFFER;
	lt->currentLen += sprintf(lt->output + lt->currentLen, "%s", lt->tempStr);
	return FFT_SUCCESS;
};

FFTResult MulComplex(FFTLayout* lt, const char* out, const char* in_1, const char* in_2, const char* temp) {
	FFTResult res = FFT_SUCCESS;
	if (strcmp(out, in_1) && strcmp(out, in_2)) {
		lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = %s.x * %s.x - %s.y * %s.y;\n\
	%s.y = %s.y * %s.x + %s.x * %s.y;\n", out, in_1, in_2, in_1, in_2, out, in_1, in_2, in_1, in_2);
	}
	else {
		if (temp) {
			lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = %s.x * %s.x - %s.y * %s.y;\n\
	%s.y = %s.y * %s.x + %s.x * %s.y;\n\
	%s = %s;\n", temp, in_1, in_2, in_1, in_2, temp, in_1, in_2, in_1, in_2, out, temp);
		}
		else
			return FFT_ERROR_NULL_TEMP_PASSED;
	}
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult SubComplex(FFTLayout* lt, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = %s.x - %s.x;\n\
	%s.y = %s.y - %s.y;\n", out, in_1, in_2, out, in_1, in_2);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult AddComplex(FFTLayout* lt, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = %s.x + %s.x;\n\
	%s.y = %s.y + %s.y;\n", out, in_1, in_2, out, in_1, in_2);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};
FFTResult AddComplexInv(FFTLayout* lt, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = - %s.x - %s.x;\n\
	%s.y = - %s.y - %s.y;\n", out, in_1, in_2, out, in_1, in_2);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult FMAComplex(FFTLayout* lt, const char* out, const char* in_1, const char* in_num, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = fma(%s.x, %s, %s.x);\n\
	%s.y = fma(%s.y, %s, %s.y);\n", out, in_1, in_num, in_2, out, in_1, in_num, in_2);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult MulComplexNumber(FFTLayout* lt, const char* out, const char* in_1, const char* in_num) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = %s.x * %s;\n\
	%s.y = %s.y * %s;\n", out, in_1, in_num, out, in_1, in_num);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};
FFTResult MovComplex(FFTLayout* lt, const char* out, const char* in) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s = %s;\n", out, in);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};
FFTResult ShuffleComplex(FFTLayout* lt, const char* out, const char* in_1, const char* in_2, const char* temp) {
	FFTResult res = FFT_SUCCESS;
	if (strcmp(out, in_2)) {
		lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = %s.x - %s.y;\n\
	%s.y = %s.y + %s.x;\n", out, in_1, in_2, out, in_1, in_2);
	}
	else {
		if (temp) {
			lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = %s.x - %s.y;\n\
	%s.y = %s.x + %s.y;\n\
	%s = %s;\n", temp, in_1, in_2, temp, in_1, in_2, out, temp);
		}
		else
			return FFT_ERROR_NULL_TEMP_PASSED;
	}
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult ShuffleComplexInv(FFTLayout* lt, const char* out, const char* in_1, const char* in_2, const char* temp) {
	FFTResult res = FFT_SUCCESS;
	if (strcmp(out, in_2)) {
		lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = %s.x + %s.y;\n\
	%s.y = %s.y - %s.x;\n", out, in_1, in_2, out, in_1, in_2);
	}
	else {
		if (temp) {
			lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = %s.x + %s.y;\n\
	%s.y = %s.x - %s.y;\n\
	%s = %s;\n", temp, in_1, in_2, temp, in_1, in_2, out, temp);
		}
		else
			return FFT_ERROR_NULL_TEMP_PASSED;
	}
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult DivComplexNumber(FFTLayout* lt, const char* out, const char* in_1, const char* in_num) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s.x = %s.x / %s;\n\
	%s.y = %s.y / %s;\n", out, in_1, in_num, out, in_1, in_num);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult AddReal(FFTLayout* lt, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s = %s + %s;\n", out, in_1, in_2);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult MovReal(FFTLayout* lt, const char* out, const char* in) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s = %s;\n", out, in);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult ModReal(FFTLayout* lt, const char* out, const char* in_1, const char* in_num) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s = %s %% %s;\n", out, in_1, in_num);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult SubReal(FFTLayout* lt, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s = %s - %s;\n", out, in_1, in_2);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult MulReal(FFTLayout* lt, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s = %s * %s;\n", out, in_1, in_2);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult SharedStore(FFTLayout* lt, const char* id, const char* in) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	sdata[%s] = %s;\n", id, in);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult SharedLoad(FFTLayout* lt, const char* out, const char* id) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "\
	%s = sdata[%s];\n", out, id);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
};


FFTResult inlineRadixKernelFFT(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t radix, uint64_t stageSize, double stageAngle, char** regID) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char LFending[4] = "";

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	char cosDef[20] = "__cosf";
	char sinDef[20] = "__sinf";
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");

	char* temp = lt->temp;
	char* w = lt->w;
	char* iw = lt->iw;
	char convolutionInverse[30] = "";
	if (lt->convolutionStep) sprintf(convolutionInverse, ", %s inverse", uintType);
	switch (radix) {
	case 2: {

		if (lt->LUT) {
			lt->tempLen = sprintf(lt->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if (!lt->inverse) {
				lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "double")) {
				lt->tempLen = sprintf(lt->tempStr, "	%s = sincos_20(angle);\n", w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(lt, temp, regID[1], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[1], regID[0], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[0], regID[0], temp);
		if (res != FFT_SUCCESS) return res;

		break;
	}
	case 3: {
		
		char* tf[2];
		for (uint64_t i = 0; i < 2; i++) {
			tf[i] = (char*)malloc(sizeof(char) * 50);
			if (!tf[i]) {
				for (uint64_t j = 0; j < i; j++) {
					free(tf[j]);
					tf[j] = 0;
				}
				return FFT_ERROR_MALLOC_FAILED;
			}
		}

		sprintf(tf[0], "-0.5%s", LFending);
		sprintf(tf[1], "-0.8660254037844386467637231707529%s", LFending);

		if (lt->LUT) {
			lt->tempLen = sprintf(lt->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if (!lt->inverse) {
				lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "double")) {
				lt->tempLen = sprintf(lt->tempStr, "	%s = sincos_20(angle*%.17f%s);\n", w, 4.0 / 3.0, LFending);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(lt, lt->locID[2], regID[2], w, 0);
		if (lt->LUT) {
			lt->tempLen = sprintf(lt->tempStr, "	%s = twiddleLUT[LUTId+%" PRIu64 "];\n", w, stageSize);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if (!lt->inverse) {
				lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {

			if (!strcmp(floatType, "double")) {
				lt->tempLen = sprintf(lt->tempStr, "	%s=sincos_20(angle*%.17f%s);\n", w, 2.0 / 3.0, LFending);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(lt, lt->locID[1], regID[1], w, 0);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(lt, regID[1], lt->locID[1], lt->locID[2]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[2], lt->locID[1], lt->locID[2]);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(lt, lt->locID[0], regID[0], regID[1]);
		if (res != FFT_SUCCESS) return res;
		res = FMAComplex(lt, lt->locID[1], regID[1], tf[0], regID[0]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, lt->locID[2], regID[2], tf[1]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, regID[0], lt->locID[0]);
		if (res != FFT_SUCCESS) return res;


		if (stageAngle < 0)
		{
			res = ShuffleComplex(lt, regID[1], lt->locID[1], lt->locID[2], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplexInv(lt, regID[2], lt->locID[1], lt->locID[2], 0);
			if (res != FFT_SUCCESS) return res;

		}
		else {
			res = ShuffleComplexInv(lt, regID[1], lt->locID[1], lt->locID[2], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplex(lt, regID[2], lt->locID[1], lt->locID[2], 0);
			if (res != FFT_SUCCESS) return res;

		}

		for (uint64_t i = 0; i < 2; i++) {
			free(tf[i]);
			tf[i] = 0;
		}
		break;
	}
	case 4: {

		if (res != FFT_SUCCESS) return res;
		if (lt->LUT) {
			lt->tempLen = sprintf(lt->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if (!lt->inverse) {
				lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {

			if (!strcmp(floatType, "double")) {
				lt->tempLen = sprintf(lt->tempStr, "	%s = sincos_20(angle);\n", w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(lt, temp, regID[2], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[2], regID[0], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[0], regID[0], temp);
		if (res != FFT_SUCCESS) return res;
		res = MulComplex(lt, temp, regID[3], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[3], regID[1], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[1], regID[1], temp);
		if (res != FFT_SUCCESS) return res;

		if (lt->LUT) {
			lt->tempLen = sprintf(lt->tempStr, "	%s=twiddleLUT[LUTId+%" PRIu64 "];\n", w, stageSize);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if (!lt->inverse) {
				lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {

			if (!strcmp(floatType, "double")) {
				lt->tempLen = sprintf(lt->tempStr, "	%s=normalize(%s + %s(1.0, 0.0));\n", w, w, vecType);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(lt, temp, regID[1], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[1], regID[0], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[0], regID[0], temp);
		if (res != FFT_SUCCESS) return res;

		if (stageAngle < 0) {
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = %s.x;", temp, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = %s.y;\n", w, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.x;\n", w, temp);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
		}
		else {
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = %s.x;", temp, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = -%s.y;\n", w, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.y = %s.x;\n", w, temp);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			//lt->tempLen = sprintf(lt->tempStr, "	w = %s(-w.y, w.x);\n\n", vecType);
		}
		res = MulComplex(lt, temp, regID[3], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[3], regID[2], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[2], regID[2], temp);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, temp, regID[1]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, regID[1], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, regID[2], temp);
		if (res != FFT_SUCCESS) return res;

		break;
	}
	case 5: {
	
		char* tf[5];
		for (uint64_t i = 0; i < 5; i++) {
			tf[i] = (char*)malloc(sizeof(char) * 50);
			if (!tf[i]) {
				for (uint64_t j = 0; j < i; j++) {
					free(tf[j]);
					tf[j] = 0;
				}
				return FFT_ERROR_MALLOC_FAILED;
			}
		}
		sprintf(tf[0], "-0.5%s", LFending);
		sprintf(tf[1], "1.538841768587626701285145288018455%s", LFending);
		sprintf(tf[2], "-0.363271264002680442947733378740309%s", LFending);
		sprintf(tf[3], "-0.809016994374947424102293417182819%s", LFending);
		sprintf(tf[4], "-0.587785252292473129168705954639073%s", LFending);

		for (uint64_t i = radix - 1; i > 0; i--) {
			if (i == radix - 1) {
				if (lt->LUT) {
					lt->tempLen = sprintf(lt->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (!lt->inverse) {
						lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {

					if (!strcmp(floatType, "double")) {
						lt->tempLen = sprintf(lt->tempStr, "	%s = sincos_20(angle*%.17f%s);\n", w, 2.0 * i / radix, LFending);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			else {
				if (lt->LUT) {
					lt->tempLen = sprintf(lt->tempStr, "	%s = twiddleLUT[LUTId+%" PRIu64 "];\n", w, (radix - 1 - i) * stageSize);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (!lt->inverse) {
						lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {

					if (!strcmp(floatType, "double")) {
						lt->tempLen = sprintf(lt->tempStr, "	%s = sincos_20(angle*%.17f%s);\n", w, 2.0 * i / radix, LFending);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			res = MulComplex(lt, lt->locID[i], regID[i], w, 0);
			if (res != FFT_SUCCESS) return res;

		}
		res = AddComplex(lt, regID[1], lt->locID[1], lt->locID[4]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[2], lt->locID[2], lt->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[3], lt->locID[2], lt->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[4], lt->locID[1], lt->locID[4]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, lt->locID[3], regID[1], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[4], regID[3], regID[4]);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(lt, lt->locID[0], regID[0], regID[1]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[0], lt->locID[0], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = FMAComplex(lt, lt->locID[1], regID[1], tf[0], regID[0]);
		if (res != FFT_SUCCESS) return res;
		res = FMAComplex(lt, lt->locID[2], regID[2], tf[0], regID[0]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, regID[3], regID[3], tf[1]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, regID[4], regID[4], tf[2]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, lt->locID[3], lt->locID[3], tf[3]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, lt->locID[4], lt->locID[4], tf[4]);
		if (res != FFT_SUCCESS) return res;

		res = SubComplex(lt, lt->locID[1], lt->locID[1], lt->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[2], lt->locID[2], lt->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[3], regID[3], lt->locID[4]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[4], lt->locID[4], regID[4]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, regID[0], lt->locID[0]);
		if (res != FFT_SUCCESS) return res;


		if (stageAngle < 0)
		{
			res = ShuffleComplex(lt, regID[1], lt->locID[1], lt->locID[4], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplex(lt, regID[2], lt->locID[2], lt->locID[3], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplexInv(lt, regID[3], lt->locID[2], lt->locID[3], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplexInv(lt, regID[4], lt->locID[1], lt->locID[4], 0);
			if (res != FFT_SUCCESS) return res;

		}
		else {
			res = ShuffleComplexInv(lt, regID[1], lt->locID[1], lt->locID[4], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplexInv(lt, regID[2], lt->locID[2], lt->locID[3], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplex(lt, regID[3], lt->locID[2], lt->locID[3], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplex(lt, regID[4], lt->locID[1], lt->locID[4], 0);
			if (res != FFT_SUCCESS) return res;

		}

		for (uint64_t i = 0; i < 5; i++) {
			free(tf[i]);
			tf[i] = 0;
		}
		break;
	}
	case 7: {
		
		char* tf[8];

		for (uint64_t i = 0; i < 8; i++) {
			tf[i] = (char*)malloc(sizeof(char) * 50);
			if (!tf[i]) {
				for (uint64_t j = 0; j < i; j++) {
					free(tf[j]);
					tf[j] = 0;
				}
				return FFT_ERROR_MALLOC_FAILED;
			}
		}
		sprintf(tf[0], "-1.16666666666666651863693004997913%s", LFending);
		sprintf(tf[1], "0.79015646852540022404554065360571%s", LFending);
		sprintf(tf[2], "0.05585426728964774240049351305970%s", LFending);
		sprintf(tf[3], "0.73430220123575240531721419756650%s", LFending);
		if (stageAngle < 0) {
			sprintf(tf[4], "0.44095855184409837868031445395900%s", LFending);
			sprintf(tf[5], "0.34087293062393136944265847887436%s", LFending);
			sprintf(tf[6], "-0.53396936033772524066165487965918%s", LFending);
			sprintf(tf[7], "0.87484229096165666561546458979137%s", LFending);
		}
		else {
			sprintf(tf[4], "-0.44095855184409837868031445395900%s", LFending);
			sprintf(tf[5], "-0.34087293062393136944265847887436%s", LFending);
			sprintf(tf[6], "0.53396936033772524066165487965918%s", LFending);
			sprintf(tf[7], "-0.87484229096165666561546458979137%s", LFending);
		}
		
		for (uint64_t i = radix - 1; i > 0; i--) {
			if (i == radix - 1) {
				if (lt->LUT) {
					lt->tempLen = sprintf(lt->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (!lt->inverse) {
						lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					if (!strcmp(floatType, "double")) {
						lt->tempLen = sprintf(lt->tempStr, "	%s = sincos_20(angle*%.17f%s);\n", w, 2.0 * i / radix, LFending);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			else {
				if (lt->LUT) {
					lt->tempLen = sprintf(lt->tempStr, "	%s = twiddleLUT[LUTId+%" PRIu64 "];\n\n", w, (radix - 1 - i) * stageSize);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (!lt->inverse) {
						lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					if (!strcmp(floatType, "double")) {
						lt->tempLen = sprintf(lt->tempStr, "	%s = sincos_20(angle*%.17f%s);\n", w, 2.0 * i / radix, LFending);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			res = MulComplex(lt, lt->locID[i], regID[i], w, 0);
			if (res != FFT_SUCCESS) return res;

		}
		res = MovComplex(lt, lt->locID[0], regID[0]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[0], lt->locID[1], lt->locID[6]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[1], lt->locID[1], lt->locID[6]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[2], lt->locID[2], lt->locID[5]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[3], lt->locID[2], lt->locID[5]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[4], lt->locID[4], lt->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[5], lt->locID[4], lt->locID[3]);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(lt, lt->locID[5], regID[1], regID[3]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[5], lt->locID[5], regID[5]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[1], regID[0], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[1], lt->locID[1], regID[4]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[0], lt->locID[0], lt->locID[1]);
		if (res != FFT_SUCCESS) return res;

		res = SubComplex(lt, lt->locID[2], regID[0], regID[4]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, lt->locID[3], regID[4], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, lt->locID[4], regID[2], regID[0]);
		if (res != FFT_SUCCESS) return res;

		res = SubComplex(lt, regID[0], regID[1], regID[5]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[2], regID[5], regID[3]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[4], regID[3], regID[1]);
		if (res != FFT_SUCCESS) return res;


		res = MulComplexNumber(lt, lt->locID[1], lt->locID[1], tf[0]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, lt->locID[2], lt->locID[2], tf[1]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, lt->locID[3], lt->locID[3], tf[2]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, lt->locID[4], lt->locID[4], tf[3]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, lt->locID[5], lt->locID[5], tf[4]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, regID[0], regID[0], tf[5]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, regID[2], regID[2], tf[6]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(lt, regID[4], regID[4], tf[7]);
		if (res != FFT_SUCCESS) return res;


		res = SubComplex(lt, regID[5], regID[4], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplexInv(lt, regID[6], regID[4], regID[0]);
		if (res != FFT_SUCCESS) return res;
		res =  AddComplex(lt, regID[4], regID[0], regID[2]);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(lt, regID[0], lt->locID[0], lt->locID[1]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[1], lt->locID[2], lt->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[2], lt->locID[4], lt->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplexInv(lt, regID[3], lt->locID[2], lt->locID[4]);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(lt, lt->locID[1], regID[0], regID[1]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[2], regID[0], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[3], regID[0], regID[3]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[4], regID[4], lt->locID[5]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[6], regID[6], lt->locID[5]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, lt->locID[5], lt->locID[5], regID[5]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, regID[0], lt->locID[0]);
		if (res != FFT_SUCCESS) return res;

		res = ShuffleComplexInv(lt, regID[1], lt->locID[1], lt->locID[4], 0);
		if (res != FFT_SUCCESS) return res;
		res = ShuffleComplexInv(lt, regID[2], lt->locID[3], lt->locID[6], 0);
		if (res != FFT_SUCCESS) return res;
		res = ShuffleComplex(lt, regID[3], lt->locID[2], lt->locID[5], 0);
		if (res != FFT_SUCCESS) return res;
		res = ShuffleComplexInv(lt, regID[4], lt->locID[2], lt->locID[5], 0);
		if (res != FFT_SUCCESS) return res;
		res = ShuffleComplex(lt, regID[5], lt->locID[3], lt->locID[6], 0);
		if (res != FFT_SUCCESS) return res;
		res = ShuffleComplex(lt, regID[6], lt->locID[1], lt->locID[4], 0);
		if (res != FFT_SUCCESS) return res;


		for (uint64_t i = 0; i < 8; i++) {
			free(tf[i]);
			tf[i] = 0;
		}
		break;
	}
	case 8: {
		
		if (lt->LUT) {
			lt->tempLen = sprintf(lt->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if (!lt->inverse) {
				lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "double")) {
				lt->tempLen = sprintf(lt->tempStr, "	%s = sincos_20(angle);\n", w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		for (uint64_t i = 0; i < 4; i++) {
			res = MulComplex(lt, temp, regID[i + 4], w, 0);
			if (res != FFT_SUCCESS) return res;
			res = SubComplex(lt, regID[i + 4], regID[i], temp);
			if (res != FFT_SUCCESS) return res;
			res = AddComplex(lt, regID[i], regID[i], temp);
			if (res != FFT_SUCCESS) return res;

		}
		if (lt->LUT) {
			lt->tempLen = sprintf(lt->tempStr, "	%s=twiddleLUT[LUTId+%" PRIu64 "];\n\n", w, stageSize);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if (!lt->inverse) {
				lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "double")) {
				lt->tempLen = sprintf(lt->tempStr, "	%s=normalize(%s + %s(1.0, 0.0));\n", w, w, vecType);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		for (uint64_t i = 0; i < 2; i++) {
			res = MulComplex(lt, temp, regID[i + 2], w, 0);
			if (res != FFT_SUCCESS) return res;
			res = SubComplex(lt, regID[i + 2], regID[i], temp);
			if (res != FFT_SUCCESS) return res;
			res = AddComplex(lt, regID[i], regID[i], temp);
			if (res != FFT_SUCCESS) return res;

		}
		if (stageAngle < 0) {
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = %s.y;\n", iw, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.x;\n", iw, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			//lt->tempLen = sprintf(lt->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = -%s.y;\n", iw, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.y = %s.x;\n", iw, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			//lt->tempLen = sprintf(lt->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}

		for (uint64_t i = 4; i < 6; i++) {
			res = MulComplex(lt, temp, regID[i + 2], iw, 0);
			if (res != FFT_SUCCESS) return res;
			res = SubComplex(lt, regID[i + 2], regID[i], temp);
			if (res != FFT_SUCCESS) return res;
			res = AddComplex(lt, regID[i], regID[i], temp);
			if (res != FFT_SUCCESS) return res;

		}

		if (lt->LUT) {
			lt->tempLen = sprintf(lt->tempStr, "	%s=twiddleLUT[LUTId+%" PRIu64 "];\n\n", w, 2 * stageSize);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if (!lt->inverse) {
				lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "double")) {
				lt->tempLen = sprintf(lt->tempStr, "	%s=normalize(%s + %s(1.0, 0.0));\n", w, w, vecType);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(lt, temp, regID[1], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[1], regID[0], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[0], regID[0], temp);
		if (res != FFT_SUCCESS) return res;

		if (stageAngle < 0) {
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = %s.y;\n", iw, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.x;\n", iw, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
		}
		else {
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = -%s.y;\n", iw, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.y = %s.x;\n", iw, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			//lt->tempLen = sprintf(lt->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		res = MulComplex(lt, temp, regID[3], iw, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[3], regID[2], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[2], regID[2], temp);
		if (res != FFT_SUCCESS) return res;
		/*lt->tempLen = sprintf(lt->tempStr, "\
temp.x = temp%s.x * iw.x - temp%s.y * iw.y;\n\
temp.y = temp%s.y * iw.x + temp%s.x * iw.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", regID[3], regID[3], regID[3], regID[3], regID[3], regID[2], regID[2], regID[2]);*/
		if (stageAngle < 0) {
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = %s.x * loc_SQRT1_2 + %s.y * loc_SQRT1_2;\n", iw, w, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.y = %s.y * loc_SQRT1_2 - %s.x * loc_SQRT1_2;\n\n", iw, w, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
		}
		else {
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = %s.x * loc_SQRT1_2 - %s.y * loc_SQRT1_2;\n", iw, w, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.y = %s.y * loc_SQRT1_2 + %s.x * loc_SQRT1_2;\n\n", iw, w, w);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
		}
		res = MulComplex(lt, temp, regID[5], iw, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[5], regID[4], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[4], regID[4], temp);
		if (res != FFT_SUCCESS) return res;
		/*lt->tempLen = sprintf(lt->tempStr, "\
temp.x = temp%s.x * iw.x - temp%s.y * iw.y;\n\
temp.y = temp%s.y * iw.x + temp%s.x * iw.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", regID[5], regID[5], regID[5], regID[5], regID[5], regID[4], regID[4], regID[4]);*/
		if (stageAngle < 0) {
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = %s.y;\n", w, iw);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.y = -%s.x;\n", w, iw);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			//lt->tempLen = sprintf(lt->tempStr, "	w = %s(iw.y, -iw.x);\n\n", vecType);
		}
		else {
			lt->tempLen = sprintf(lt->tempStr, "	%s.x = -%s.y;\n", w, iw);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			lt->tempLen = sprintf(lt->tempStr, "	%s.y = %s.x;\n", w, iw);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			//lt->tempLen = sprintf(lt->tempStr, "	w = %s(-iw.y, iw.x);\n\n", vecType);
		}
		res = MulComplex(lt, temp, regID[7], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(lt, regID[7], regID[6], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(lt, regID[6], regID[6], temp);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, temp, regID[1]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, regID[1], regID[4]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, regID[4], temp);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, temp, regID[3]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, regID[3], regID[6]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(lt, regID[6], temp);
		if (res != FFT_SUCCESS) return res;

		break;
	}


	}
	return res;
};


FFTResult appendExtensions(FFTLayout* lt, const char* floatType, const char* floatTypeInputMemory, const char* floatTypeOutputMemory, const char* floatTypeKernelMemory) {
	FFTResult res = FFT_SUCCESS;

	lt->tempLen = sprintf(lt->tempStr, "\
#include <hip/hip_runtime.h>\n");
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;

	return res;
}
FFTResult appendPushConstant(FFTLayout* lt, const char* type, const char* name) {
	FFTResult res = FFT_SUCCESS;
	lt->tempLen = sprintf(lt->tempStr, "	%s %s;\n", type, name);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
}

FFTResult appendConstant(FFTLayout* lt, const char* type, const char* name, const char* defaultVal, const char* LFending) {
	FFTResult res = FFT_SUCCESS;

	lt->tempLen = sprintf(lt->tempStr, "const %s %s = %s%s;\n", type, name, defaultVal, LFending);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
}

FFTResult AppendLineFromInput(FFTLayout* lt, const char* in) {
	//appends code line stored in tempStr to generated code
	if (lt->currentLen + (int64_t)strlen(in) > lt->maxCodeLength) return FFT_ERROR_INSUFFICIENT_CODE_BUFFER;
	lt->currentLen += sprintf(lt->output + lt->currentLen, "%s", in);
	return FFT_SUCCESS;
};

FFTResult appendConstantsFFT(FFTLayout* lt, const char* floatType, const char* uintType) {
	FFTResult res = FFT_SUCCESS;
	char LFending[4] = "";
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");

	res = appendConstant(lt, floatType, "loc_PI", "3.1415926535897932384626433832795", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "loc_SQRT1_2", "0.70710678118654752440084436210485", LFending);
	if (res != FFT_SUCCESS) return res;
	return res;
}




FFTResult appendSinCos20(FFTLayout* lt, const char* floatType, const char* uintType) {
	FFTResult res = FFT_SUCCESS;
	char functionDefinitions[100] = "";
	char vecType[30];
	char LFending[4] = "";

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");
	sprintf(functionDefinitions, "__device__ static __inline__ ");

	res = appendConstant(lt, floatType, "loc_2_PI", "0.63661977236758134307553505349006", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "loc_PI_2", "1.5707963267948966192313216916398", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "a1", "0.99999999999999999999962122687403772", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "a3", "-0.166666666666666666637194166219637268", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "a5", "0.00833333333333333295212653322266277182", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "a7", "-0.000198412698412696489459896530659927773", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "a9", "2.75573192239364018847578909205399262e-6", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "a11", "-2.50521083781017605729370231280411712e-8", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "a13", "1.60590431721336942356660057796782021e-10", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "a15", "-7.64712637907716970380859898835680587e-13", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "a17", "2.81018528153898622636194976499656274e-15", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(lt, floatType, "ab", "-7.97989713648499642889739108679114937e-18", LFending);
	if (res != FFT_SUCCESS) return res;
	lt->tempLen = sprintf(lt->tempStr, "\
%s%s sincos_20(double x)\n\
{\n\
	//minimax coefs for sin for 0..pi/2 range\n\
	double y = abs(x * loc_2_PI);\n\
	double q = floor(y);\n\
	int quadrant = int(q);\n\
	double t = (quadrant & 1) != 0 ? 1 - y + q : y - q;\n\
	t *= loc_PI_2;\n\
	double t2 = t * t;\n\
	double r = fma(fma(fma(fma(fma(fma(fma(fma(fma(ab, t2, a17), t2, a15), t2, a13), t2, a11), t2, a9), t2, a7), t2, a5), t2, a3), t2 * t, t);\n\
	%s cos_sin;\n\
	cos_sin.x = ((quadrant == 0) || (quadrant == 3)) ? sqrt(1 - r * r) : -sqrt(1 - r * r);\n\
	r = x < 0 ? -r : r;\n\
	cos_sin.y = (quadrant & 2) != 0 ? -r : r;\n\
	return cos_sin;\n\
}\n\n", functionDefinitions, vecType, vecType);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
}

FFTResult appendConversion(FFTLayout* lt, const char* floatType, const char* floatTypeDifferent) {
	FFTResult res = FFT_SUCCESS;

	char functionDefinitions[100] = "";
	char vecType[30];
	char vecTypeDifferent[30];
	sprintf(functionDefinitions, "__device__ static __inline__ ");

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");

	if (!strcmp(floatTypeDifferent, "double")) sprintf(vecTypeDifferent, "double2");
	lt->tempLen = sprintf(lt->tempStr, "\
%s%s conv_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x = (%s) input.x;\n\
	ret_val.y = (%s) input.y;\n\
	return ret_val;\n\
}\n\n", functionDefinitions, vecType, vecType, vecTypeDifferent, vecType, floatType, floatType);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	lt->tempLen = sprintf(lt->tempStr, "\
%s%s conv_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x = (%s) input.x;\n\
	ret_val.y = (%s) input.y;\n\
	return ret_val;\n\
}\n\n", functionDefinitions, vecTypeDifferent, vecTypeDifferent, vecType, vecTypeDifferent, floatTypeDifferent, floatTypeDifferent);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	return res;
}



FFTResult appendPushConstantsFFT(FFTLayout* lt, const char* floatType, const char* uintType) {
	FFTResult res = FFT_SUCCESS;

	lt->tempLen = sprintf(lt->tempStr, "	typedef struct {\n");
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	res = appendPushConstant(lt, uintType, "coordinate");
	if (res != FFT_SUCCESS) return res;
	res = appendPushConstant(lt, uintType, "batchID");
	if (res != FFT_SUCCESS) return res;
	res = appendPushConstant(lt, uintType, "workGroupShiftX");
	if (res != FFT_SUCCESS) return res;
	res = appendPushConstant(lt, uintType, "workGroupShiftY");
	if (res != FFT_SUCCESS) return res;
	res = appendPushConstant(lt, uintType, "workGroupShiftZ");
	if (res != FFT_SUCCESS) return res;
	lt->tempLen = sprintf(lt->tempStr, "	}PushConsts;\n");
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	lt->tempLen = sprintf(lt->tempStr, "	__constant__ PushConsts consts;\n");
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;

	return res;
}


FFTResult appendInputLayoutFFT(FFTLayout* lt, uint64_t id, const char* floatTypeMemory, uint64_t inputType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	switch (inputType) {
	case 0: case 1: case 2: case 3: case 4: case 6: {
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double2");
		break;
	}
	case 5: case 120: case 121: case 130: case 131: case 140: case 141: case 142: case 143:
	{
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double");
		break;
	}
	}
	return res;
}

FFTResult appendOutputLayoutFFT(FFTLayout* lt, uint64_t id, const char* floatTypeMemory, uint64_t outputType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	switch (outputType) {
	case 0: case 1: case 2: case 3: case 4: case 5: {
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double2");
		break;
	}
	case 6: case 120: case 121: case 130: case 131: case 140: case 141: case 142: case 143:
	{
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double");
		break;
	}
	}
	return res;
}

FFTResult appendLUTLayoutFFT(FFTLayout* lt, uint64_t id, const char* floatType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	return res;
}

FFTResult appendSharedMemoryFFT(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t sharedType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char sharedDefinitions[20] = "";
	uint64_t vecSize = 1;
	uint64_t maxSequenceSharedMemory = 0;

	if (!strcmp(floatType, "double")) {
		sprintf(vecType, "double2");
		sprintf(sharedDefinitions, "__shared__");
		vecSize = 16;
	}
	maxSequenceSharedMemory = lt->sharedMemSize / vecSize;
	//maxSequenceSharedMemoryPow2 = lt->sharedMemSizePow2 / vecSize;
	uint64_t mergeR2C = (lt->mergeSequencesR2C && (lt->axis_id == 0)) ? 2 : 0;
	switch (sharedType) {
	case 0: case 5: case 6: case 120: case 130: case 140: case 142:
	{
		lt->conflictStages = 0;
		lt->conflictStride = ((lt->dim > lt->numSharedBanks / 2) && ((lt->dim & (lt->dim - 1)) == 0)) ? lt->dim / lt->regAd * (lt->numSharedBanks / 2 + 1) / (lt->numSharedBanks / 2) : lt->dim / lt->regAd;
		lt->conflictShared = ((lt->numSharedBanks / 2 <= lt->localSize[1])) ? lt->dim / lt->regAd + 1 : lt->dim / lt->regAd + (lt->numSharedBanks / 2) / lt->localSize[1];
		if (lt->conflictShared < lt->dim / lt->regAd + mergeR2C) lt->conflictShared = lt->dim / lt->regAd + mergeR2C;
		lt->maxSharedStride = (lt->conflictStride < lt->conflictShared) ? lt->conflictShared : lt->conflictStride;


		lt->usedSharedMemory = vecSize * lt->localSize[1] * lt->maxSharedStride;
		lt->maxSharedStride = ((lt->sharedMemSize < lt->usedSharedMemory)) ? lt->dim / lt->regAd : lt->maxSharedStride;

		lt->conflictStride = (lt->maxSharedStride == lt->dim / lt->regAd) ? lt->dim / lt->regAd : lt->conflictStride;
		lt->conflictShared = (lt->maxSharedStride == lt->dim / lt->regAd) ? lt->dim / lt->regAd : lt->conflictShared;

		lt->tempLen = sprintf(lt->tempStr, "%s sharedStride = %" PRIu64 ";\n", uintType, lt->conflictShared);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;

		lt->tempLen = sprintf(lt->tempStr, "%s* sdata = (%s*)shared;\n\n", vecType, vecType);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		lt->usedSharedMemory = vecSize * lt->localSize[1] * lt->maxSharedStride;
		break;
	}
	case 1: case 2: case 121: case 131: case 141: case 143:
	{
		uint64_t shift = (lt->dim < (lt->numSharedBanks / 2)) ? (lt->numSharedBanks / 2) / lt->dim : 1;
		lt->conflictShared = ((lt->axisSwapped) && ((lt->localSize[0] % 4) == 0)) ? lt->localSize[0] + shift : lt->localSize[0];
		lt->maxSharedStride = ((maxSequenceSharedMemory < lt->conflictShared* lt->dim / lt->regAd)) ? lt->localSize[0] : lt->conflictShared;

		lt->conflictShared = (lt->maxSharedStride == lt->localSize[0]) ? lt->localSize[0] : lt->conflictShared;
		lt->tempLen = sprintf(lt->tempStr, "%s sharedStride = %" PRIu64 ";\n", uintType, lt->maxSharedStride);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;


		lt->tempLen = sprintf(lt->tempStr, "%s* sdata = (%s*)shared;\n\n", vecType, vecType);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		lt->usedSharedMemory = vecSize * lt->maxSharedStride * (lt->dim + mergeR2C) / lt->regAd;

		break;
	}
	}
	return res;
}

FFTResult appendInitialization(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t initType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");

	uint64_t threadStorage = lt->threadRegister * lt->regAd;
	uint64_t threadRegister = lt->threadRegister;

	for (uint64_t i = 0; i < lt->threadRegister; i++) {
		lt->tempLen = sprintf(lt->tempStr, "	%s temp_%" PRIu64 ";\n", vecType, i);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	lt->regIDs = (char**)malloc(sizeof(char*) * threadStorage);
	if (!lt->regIDs) return FFT_ERROR_MALLOC_FAILED;
	for (uint64_t i = 0; i < threadStorage; i++) {
		lt->regIDs[i] = (char*)malloc(sizeof(char) * 50);
		if (!lt->regIDs[i]) {
			for (uint64_t j = 0; j < i; j++) {
				free(lt->regIDs[j]);
				lt->regIDs[j] = 0;
			}
			free(lt->regIDs);
			lt->regIDs = 0;
			return FFT_ERROR_MALLOC_FAILED;
		}
		if (i < threadRegister)
			sprintf(lt->regIDs[i], "temp_%" PRIu64 "", i);
		else
			sprintf(lt->regIDs[i], "temp_%" PRIu64 "", i);
	

	}
	if (lt->regAd > 1) {
		
		for (uint64_t i = 1; i < lt->regAd; i++) {
			for (uint64_t j = 0; j < lt->threadRegister; j++) {
				lt->tempLen = sprintf(lt->tempStr, "	%s temp_%" PRIu64 ";\n", vecType, j + i * lt->threadRegister);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}

		}
	}
	lt->tempLen = sprintf(lt->tempStr, "	%s w;\n", vecType);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	sprintf(lt->w, "w");
	uint64_t maxNonPow2Radix = 1;
	if (lt->dim % 3 == 0) maxNonPow2Radix = 3;
	if (lt->dim % 5 == 0) maxNonPow2Radix = 5;
	if (lt->dim % 7 == 0) maxNonPow2Radix = 7;
	if (lt->dim % 11 == 0) maxNonPow2Radix = 11;
	if (lt->dim % 13 == 0) maxNonPow2Radix = 13;
	for (uint64_t i = 0; i < maxNonPow2Radix; i++) {
		sprintf(lt->locID[i], "loc_%" PRIu64 "", i);
		lt->tempLen = sprintf(lt->tempStr, "	%s %s;\n", vecType, lt->locID[i]);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	sprintf(lt->temp, "%s", lt->locID[0]);
	uint64_t useRadix8 = 0;
	for (uint64_t i = 0; i < lt->numStages; i++)
		if (lt->stageRadix[i] == 8) useRadix8 = 1;
	if (useRadix8 == 1) {
		if (maxNonPow2Radix > 1) sprintf(lt->iw, "%s", lt->locID[1]);
		else {
			lt->tempLen = sprintf(lt->tempStr, "	%s iw;\n", vecType);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			sprintf(lt->iw, "iw");
		}
	}
	//lt->tempLen = sprintf(lt->tempStr, "	%s %s;\n", vecType, lt->tempReg);
	lt->tempLen = sprintf(lt->tempStr, "	%s %s;\n", uintType, lt->stageInvocationID);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	lt->tempLen = sprintf(lt->tempStr, "	%s %s;\n", uintType, lt->blockInvocationID);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	lt->tempLen = sprintf(lt->tempStr, "	%s %s;\n", uintType, lt->sdataID);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	lt->tempLen = sprintf(lt->tempStr, "	%s %s;\n", uintType, lt->combinedID);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	lt->tempLen = sprintf(lt->tempStr, "	%s %s;\n", uintType, lt->inoutID);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;
	if (lt->LUT) {
		lt->tempLen = sprintf(lt->tempStr, "	%s LUTId=0;\n", uintType);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	else {
		lt->tempLen = sprintf(lt->tempStr, "	%s angle=0;\n", floatType);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	if (((lt->stageStartSize > 1) && (!((lt->stageStartSize > 1) && (!lt->reorderFourStep) && (lt->inverse)))) || (((lt->stageStartSize > 1) && (!lt->reorderFourStep) && (lt->inverse))) || (0)) {
		lt->tempLen = sprintf(lt->tempStr, "	%s mult;\n", vecType);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		lt->tempLen = sprintf(lt->tempStr, "	mult.x = 0;\n");
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		lt->tempLen = sprintf(lt->tempStr, "	mult.y = 0;\n");
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	return res;
}

FFTResult appendBarrier(FFTLayout* lt, uint64_t numTab) {
	FFTResult res = FFT_SUCCESS;
	char tabs[100];
	for (uint64_t i = 0; i < numTab; i++)
		sprintf(tabs, "	");

	lt->tempLen = sprintf(lt->tempStr, "%s__syncthreads();\n\n", tabs);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) return res;

	return res;
}


FFTResult threadDataOrder(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t shuffleType, uint64_t start) {
	FFTResult res = FFT_SUCCESS;
	switch (shuffleType) {
	case 0: case 5: case 6: case 120: case 130: case 140: case 142: {
		uint64_t threadStorage;
		if (start == 1) {
			threadStorage = lt->threadRadixRegister[lt->stageRadix[0]] * lt->regAd;
		}
		else {
			threadStorage = lt->threadRadixRegister[lt->stageRadix[lt->numStages - 1]] * lt->regAd;
		}
		uint64_t logicalGroupSize = lt->dim / threadStorage;
		if ((lt->regAd > 1) && (threadStorage != lt->threadRegisterMin * lt->regAd)) {
			for (uint64_t k = 0; k < lt->regAd; k++) {
				if (k > 0) {
					res = appendBarrier(lt, 2);
					if (res != FFT_SUCCESS) return res;
				}

				res = AppendLineFromInput(lt, lt->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (start == 0) {
					lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_x, threadStorage, lt->dim);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					for (uint64_t i = 0; i < threadStorage / lt->regAd; i++) {
						lt->tempLen = sprintf(lt->tempStr, "\
	sdata[%s + %" PRIu64 "] = %s;\n", lt->gl_LocalInvocationID_x, i * logicalGroupSize, lt->regIDs[i + k * lt->threadRegister]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
					lt->tempLen = sprintf(lt->tempStr, "	}\n");
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else
				{
					for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
						lt->tempLen = sprintf(lt->tempStr, "\
	sdata[%s + %" PRIu64 "] = %s;\n", lt->gl_LocalInvocationID_x, i * lt->localSize[0], lt->regIDs[i + k * lt->threadRegister]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				res = AppendLineFromInput(lt, lt->disableThreadsEnd);
				if (res != FFT_SUCCESS) return res;
				res = appendBarrier(lt, 2);
				if (res != FFT_SUCCESS) return res;
				res = AppendLineFromInput(lt, lt->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (start == 1) {
					lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_x, threadStorage, lt->dim);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					for (uint64_t i = 0; i < threadStorage / lt->regAd; i++) {
						lt->tempLen = sprintf(lt->tempStr, "\
	%s = sdata[%s + %" PRIu64 "];\n", lt->regIDs[i + k * lt->threadRegister], lt->gl_LocalInvocationID_x, i * logicalGroupSize);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
					lt->tempLen = sprintf(lt->tempStr, "	}\n");
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
						lt->tempLen = sprintf(lt->tempStr, "\
	%s = sdata[%s + %" PRIu64 "];\n", lt->regIDs[i + k * lt->threadRegister], lt->gl_LocalInvocationID_x, i * lt->localSize[0]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				res = AppendLineFromInput(lt, lt->disableThreadsEnd);
				if (res != FFT_SUCCESS) return res;
			}
		}

		break;
	}
	case 1: case 2: case 121: case 131: case 141: case 143: {
		uint64_t threadStorage;
		if (start == 1) {
			threadStorage = lt->threadRadixRegister[lt->stageRadix[0]] * lt->regAd;// (lt->threadRegister % lt->stageRadix[0] == 0) ? lt->threadRegister * lt->regAd : lt->threadRegisterMin * lt->regAd;
		}
		else {
			threadStorage = lt->threadRadixRegister[lt->stageRadix[lt->numStages - 1]] * lt->regAd;// (lt->threadRegister % lt->stageRadix[lt->numStages - 1] == 0) ? lt->threadRegister * lt->regAd : lt->threadRegisterMin * lt->regAd;
		}
		uint64_t logicalGroupSize = lt->dim / threadStorage;
		if ((lt->regAd > 1) && (threadStorage != lt->threadRegisterMin * lt->regAd)) {
			for (uint64_t k = 0; k < lt->regAd; k++) {
				if (k > 0) {
					res = appendBarrier(lt, 2);
					if (res != FFT_SUCCESS) return res;
				}
				res = AppendLineFromInput(lt, lt->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (start == 0) {
					lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_y, threadStorage, lt->dim);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					for (uint64_t i = 0; i < threadStorage / lt->regAd; i++) {
						lt->tempLen = sprintf(lt->tempStr, "\
	sdata[%s + %s * (%s + %" PRIu64 ")] = %s;\n", lt->gl_LocalInvocationID_x, lt->sharedStride, lt->gl_LocalInvocationID_y, i * logicalGroupSize, lt->regIDs[i + k * lt->threadRegister]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
					lt->tempLen = sprintf(lt->tempStr, "	}\n");
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else
				{
					for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
						lt->tempLen = sprintf(lt->tempStr, "\
	sdata[%s + %s * (%s + %" PRIu64 ")] = %s;\n", lt->gl_LocalInvocationID_x, lt->sharedStride, lt->gl_LocalInvocationID_y, i * lt->localSize[1], lt->regIDs[i + k * lt->threadRegister]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				res = AppendLineFromInput(lt, lt->disableThreadsEnd);
				if (res != FFT_SUCCESS) return res;
				res = appendBarrier(lt, 2);
				if (res != FFT_SUCCESS) return res;
				res = AppendLineFromInput(lt, lt->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (start == 1) {
					lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_y, threadStorage, lt->dim);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					for (uint64_t i = 0; i < threadStorage / lt->regAd; i++) {
						lt->tempLen = sprintf(lt->tempStr, "\
	%s = sdata[%s + %s * (%s + %" PRIu64 ")];\n", lt->regIDs[i + k * lt->threadRegister], lt->gl_LocalInvocationID_x, lt->sharedStride, lt->gl_LocalInvocationID_y, i * logicalGroupSize);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
					lt->tempLen = sprintf(lt->tempStr, "	}\n");
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
						lt->tempLen = sprintf(lt->tempStr, "\
	%s = sdata[%s + %s * (%s + %" PRIu64 ")];\n", lt->regIDs[i + k * lt->threadRegister], lt->gl_LocalInvocationID_x, lt->sharedStride, lt->gl_LocalInvocationID_y, i * lt->localSize[1]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				res = AppendLineFromInput(lt, lt->disableThreadsEnd);
				if (res != FFT_SUCCESS) return res;
			}
		}

		break;
	}
	}
	return res;
}



FFTResult radixNonStrided(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char LFending[4] = "";

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");


	char convolutionInverse[10] = "";
	if (lt->convolutionStep) {
		if (stageAngle < 0)
			sprintf(convolutionInverse, ", 0");
		else
			sprintf(convolutionInverse, ", 1");
	}
	uint64_t threadStorage = lt->threadRadixRegister[stageRadix] * lt->regAd;
	uint64_t threadRegister = lt->threadRadixRegister[stageRadix];
	uint64_t logicalGroupSize = lt->dim / threadStorage;
	if ((lt->localSize[0] * threadStorage > lt->dim) || (stageSize > 1) || ((lt->localSize[1] > 1) && (!(0 && (stageAngle > 0)))) || ((lt->convolutionStep) && ((lt->matrixConvolution > 1) || (lt->numKernels > 1)) && (stageAngle > 0)) || (0))
	{
		res = appendBarrier(lt, 1);
		if (res != FFT_SUCCESS) return res;
	}
	res = AppendLineFromInput(lt, lt->disableThreadsStart);
	if (res != FFT_SUCCESS) return res;

	if (lt->localSize[0] * threadStorage > lt->dim) {
		lt->tempLen = sprintf(lt->tempStr, "\
		if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_LocalInvocationID_x, threadStorage, lt->dim);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	for (uint64_t k = 0; k < lt->regAd; k++) {
		for (uint64_t j = 0; j < threadRegister / stageRadix; j++) {
			lt->tempLen = sprintf(lt->tempStr, "\
		%s = (%s+ %" PRIu64 ") %% (%" PRIu64 ");\n", lt->stageInvocationID, lt->gl_LocalInvocationID_x, (j + k * threadRegister / stageRadix) * logicalGroupSize, stageSize);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if (lt->LUT)
				lt->tempLen = sprintf(lt->tempStr, "		LUTId = stageInvocationID + %" PRIu64 ";\n", stageSizeSum);
			else
				lt->tempLen = sprintf(lt->tempStr, "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if ((lt->regAd == 1) && ((lt->localSize[0] * threadStorage > lt->dim) || (stageSize > 1) || ((lt->localSize[1] > 1) && (!(0 && (stageAngle > 0)))) || ((lt->convolutionStep) && ((lt->matrixConvolution > 1) || (lt->numKernels > 1)) && (stageAngle > 0)) || 0)) {
				for (uint64_t i = 0; i < stageRadix; i++) {
					uint64_t id = j + i * threadRegister / stageRadix;
					id = (id / threadRegister) * lt->threadRegister + id % threadRegister;

					lt->tempLen = sprintf(lt->tempStr, "\
		%s = %s + %" PRIu64 ";\n", lt->sdataID, lt->gl_LocalInvocationID_x, j * logicalGroupSize + i * lt->dim / stageRadix);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					if (lt->conflictStages == 1) {
						lt->tempLen = sprintf(lt->tempStr, "\
	%s = (%s / %" PRIu64 ") * %" PRIu64 " + %s %% %" PRIu64 ";", lt->sdataID, lt->sdataID, lt->numSharedBanks / 2, lt->numSharedBanks / 2 + 1, lt->sdataID, lt->numSharedBanks / 2);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}

					if (lt->localSize[1] > 1) {
						lt->tempLen = sprintf(lt->tempStr, "\
		%s = %s + sharedStride * %s;\n", lt->sdataID, lt->sdataID, lt->gl_LocalInvocationID_y);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
					lt->tempLen = sprintf(lt->tempStr, "\
		%s = sdata[%s];\n", lt->regIDs[id], lt->sdataID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
			}
			char** regID = (char**)malloc(sizeof(char*) * stageRadix);
			if (regID) {
				for (uint64_t i = 0; i < stageRadix; i++) {
					regID[i] = (char*)malloc(sizeof(char) * 50);
					if (!regID[i]) {
						for (uint64_t j = 0; j < i; j++) {
							free(regID[j]);
							regID[j] = 0;
						}
						free(regID);
						regID = 0;
						return FFT_ERROR_MALLOC_FAILED;
					}
					uint64_t id = j + k * threadRegister / stageRadix + i * threadStorage / stageRadix;
					id = (id / threadRegister) * lt->threadRegister + id % threadRegister;
					sprintf(regID[i], "%s", lt->regIDs[id]);

				}
				res = inlineRadixKernelFFT(lt, floatType, uintType, stageRadix, stageSize, stageAngle, regID);
				if (res != FFT_SUCCESS) return res;
				for (uint64_t i = 0; i < stageRadix; i++) {
					uint64_t id = j + k * threadRegister / stageRadix + i * threadStorage / stageRadix;
					id = (id / threadRegister) * lt->threadRegister + id % threadRegister;
					sprintf(lt->regIDs[id], "%s", regID[i]);
				}
				for (uint64_t i = 0; i < stageRadix; i++) {
					free(regID[i]);
					regID[i] = 0;
				}
				free(regID);
				regID = 0;
			}
			else
				return FFT_ERROR_MALLOC_FAILED;
		}
	}
	if (lt->localSize[0] * threadStorage > lt->dim) {
		lt->tempLen = sprintf(lt->tempStr, "		}\n");
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	res = AppendLineFromInput(lt, lt->disableThreadsEnd);
	if (res != FFT_SUCCESS) return res;
	return res;
}
FFTResult radixStrided(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char LFending[4] = "";

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");


	char convolutionInverse[10] = "";
	if (lt->convolutionStep) {
		if (stageAngle < 0)
			sprintf(convolutionInverse, ", 0");
		else
			sprintf(convolutionInverse, ", 1");
	}
	uint64_t threadStorage = lt->threadRadixRegister[stageRadix] * lt->regAd;// (lt->threadRegister % stageRadix == 0) ? lt->threadRegister * lt->regAd : lt->threadRegisterMin * lt->regAd;
	uint64_t threadRegister = lt->threadRadixRegister[stageRadix];// (lt->threadRegister % stageRadix == 0) ? lt->threadRegister : lt->threadRegisterMin;
	uint64_t logicalGroupSize = lt->dim / threadStorage;
	if (((lt->axis_id == 0) && (lt->axis_upload_id == 0) && (!(0 && (stageAngle > 0)))) || (lt->localSize[1] * threadStorage > lt->dim) || (stageSize > 1) || ((lt->convolutionStep) && ((lt->matrixConvolution > 1) || (lt->numKernels > 1)) && (stageAngle > 0)) || (0))
	{
		res = appendBarrier(lt, 1);
		if (res != FFT_SUCCESS) return res;
	}
	res = AppendLineFromInput(lt, lt->disableThreadsStart);
	if (res != FFT_SUCCESS) return res;
	if (lt->localSize[1] * threadStorage > lt->dim) {
		lt->tempLen = sprintf(lt->tempStr, "\
		if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_LocalInvocationID_y, threadStorage, lt->dim);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	for (uint64_t k = 0; k < lt->regAd; k++) {
		for (uint64_t j = 0; j < threadRegister / stageRadix; j++) {
			lt->tempLen = sprintf(lt->tempStr, "\
		%s = (%s+ %" PRIu64 ") %% (%" PRIu64 ");\n", lt->stageInvocationID, lt->gl_LocalInvocationID_y, (j + k * threadRegister / stageRadix) * logicalGroupSize, stageSize);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if (lt->LUT)
				lt->tempLen = sprintf(lt->tempStr, "		LUTId = stageInvocationID + %" PRIu64 ";\n", stageSizeSum);
			else
				lt->tempLen = sprintf(lt->tempStr, "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
			if ((lt->regAd == 1) && (((lt->axis_id == 0) && (lt->axis_upload_id == 0) && (!(0 && (stageAngle > 0)))) || (lt->localSize[1] * threadStorage > lt->dim) || (stageSize > 1) || ((lt->convolutionStep) && ((lt->matrixConvolution > 1) || (lt->numKernels > 1)) && (stageAngle > 0)) || (0))) {
				for (uint64_t i = 0; i < stageRadix; i++) {
					uint64_t id = j + i * threadRegister / stageRadix;
					id = (id / threadRegister) * lt->threadRegister + id % threadRegister;
					lt->tempLen = sprintf(lt->tempStr, "\
		%s = sdata[%s*(%s+%" PRIu64 ")+%s];\n", lt->regIDs[id], lt->sharedStride, lt->gl_LocalInvocationID_y, j * logicalGroupSize + i * lt->dim / stageRadix, lt->gl_LocalInvocationID_x);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
			}

			char** regID = (char**)malloc(sizeof(char*) * stageRadix);
			if (regID) {
				for (uint64_t i = 0; i < stageRadix; i++) {
					regID[i] = (char*)malloc(sizeof(char) * 50);
					if (!regID[i]) {
						for (uint64_t j = 0; j < i; j++) {
							free(regID[j]);
							regID[j] = 0;
						}
						free(regID);
						regID = 0;
						return FFT_ERROR_MALLOC_FAILED;
					}
					uint64_t id = j + k * threadRegister / stageRadix + i * threadStorage / stageRadix;
					id = (id / threadRegister) * lt->threadRegister + id % threadRegister;
					sprintf(regID[i], "%s", lt->regIDs[id]);
					/*if (j + i * threadStorage / stageRadix < threadRegister)
						sprintf(regID[i], "_%" PRIu64 "", j + i * threadStorage / stageRadix);
					else
						sprintf(regID[i], "%" PRIu64 "[%" PRIu64 "]", (j + i * threadStorage / stageRadix) / threadRegister, (j + i * threadStorage / stageRadix) % threadRegister);*/

				}
				res = inlineRadixKernelFFT(lt, floatType, uintType, stageRadix, stageSize, stageAngle, regID);
				if (res != FFT_SUCCESS) return res;
				for (uint64_t i = 0; i < stageRadix; i++) {
					uint64_t id = j + k * threadRegister / stageRadix + i * threadStorage / stageRadix;
					id = (id / threadRegister) * lt->threadRegister + id % threadRegister;
					sprintf(lt->regIDs[id], "%s", regID[i]);
				}
				for (uint64_t i = 0; i < stageRadix; i++) {
					free(regID[i]);
					regID[i] = 0;
				}
				free(regID);
				regID = 0;
			}
			else
				return FFT_ERROR_MALLOC_FAILED;
		}
	}
	if (lt->localSize[1] * threadStorage > lt->dim) {
		lt->tempLen = sprintf(lt->tempStr, "		}\n");
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	res = AppendLineFromInput(lt, lt->disableThreadsEnd);
	if (res != FFT_SUCCESS) return res;
	if (stageSize == 1) {
		lt->tempLen = sprintf(lt->tempStr, "		%s = %" PRIu64 ";\n", lt->sharedStride, lt->localSize[0]);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	return res;
}


FFTResult appendRadixStage(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t shuffleType) {
	FFTResult res = FFT_SUCCESS;
	switch (shuffleType) {
	case 0: case 5: case 6: case 120: case 130: case 140: case 142: {
		res = radixNonStrided(lt, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 1: case 2: case 121: case 131: case 141: case 143: {
		res = radixStrided(lt, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	}
	return res;
}


FFTResult appendregAdShuffle(FFTLayout* lt, const char* floatType, uint64_t stageSize, uint64_t stageRadixPrev, uint64_t stageRadix, double stageAngle) {
	FFTResult res = FFT_SUCCESS;
	if (((lt->actualInverse) && (lt->normalize)) || ((lt->convolutionStep) && (stageAngle > 0))) {
		char stageNormalization[10] = "";
		if ((stageSize == 1) && (0)) {
			if (0 == 4)
				sprintf(stageNormalization, "%" PRIu64 "", stageRadixPrev * stageRadix * 4);
			else
				sprintf(stageNormalization, "%" PRIu64 "", stageRadixPrev * stageRadix * 2);
		}
		else
			sprintf(stageNormalization, "%" PRIu64 "", stageRadixPrev * stageRadix);
		uint64_t threadRegister = lt->threadRadixRegister[stageRadix];// (lt->threadRegister % stageRadix == 0) ? lt->threadRegister : lt->threadRegisterMin;
		for (uint64_t k = 0; k < lt->regAd; ++k) {
			for (uint64_t i = 0; i < threadRegister; i++) {
				res = DivComplexNumber(lt, lt->regIDs[i + k * lt->threadRegister], lt->regIDs[i + k * lt->threadRegister], stageNormalization);
				if (res != FFT_SUCCESS) return res;
			}
		}
	}
	return res;
}


FFTResult appendRadixShuffleNonStrided(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");

	char stageNormalization[10] = "";
	if (((lt->actualInverse) && (lt->normalize)) || ((lt->convolutionStep) && (stageAngle > 0))) {
		if ((stageSize == 1) && (0)) {
			if (0 == 4)
				sprintf(stageNormalization, "%" PRIu64 "", stageRadix * 4);
			else
				sprintf(stageNormalization, "%" PRIu64 "", stageRadix * 2);
		}
		else
			sprintf(stageNormalization, "%" PRIu64 "", stageRadix);
	}

	char tempNum[50] = "";

	uint64_t threadStorage = lt->threadRadixRegister[stageRadix] * lt->regAd;// (lt->threadRegister % stageRadix == 0) ? lt->threadRegister * lt->regAd : lt->threadRegisterMin * lt->regAd;
	uint64_t threadStorageNext = lt->threadRadixRegister[stageRadixNext] * lt->regAd;// (lt->threadRegister % stageRadixNext == 0) ? lt->threadRegister * lt->regAd : lt->threadRegisterMin * lt->regAd;
	uint64_t threadRegister = lt->threadRadixRegister[stageRadix];// (lt->threadRegister % stageRadix == 0) ? lt->threadRegister : lt->threadRegisterMin;
	uint64_t threadRegisterNext = lt->threadRadixRegister[stageRadixNext];// (lt->threadRegister % stageRadixNext == 0) ? lt->threadRegister : lt->threadRegisterMin;

	uint64_t logicalGroupSize = lt->dim / threadStorage;
	uint64_t logicalGroupSizeNext = lt->dim / threadStorageNext;
	if (((lt->regAd == 1) && ((lt->localSize[0] * threadStorage > lt->dim) || (stageSize < lt->dim / stageRadix) || ((lt->reorderFourStep) && (lt->dim < lt->fft_dim_full) && (lt->localSize[1] > 1)) || (lt->localSize[1] > 1) || (0 && (!lt->inverse) && (lt->axis_id == 0)) || ((lt->convolutionStep) && ((lt->matrixConvolution > 1) || (lt->numKernels > 1)) && (stageAngle < 0)))) || (0))
	{
		res = appendBarrier(lt, 1);
		if (res != FFT_SUCCESS) return res;
	}
	if ((lt->localSize[0] * threadStorage > lt->dim) || (stageSize < lt->dim / stageRadix) || ((lt->reorderFourStep) && (lt->dim < lt->fft_dim_full) && (lt->localSize[1] > 1)) || (lt->localSize[1] > 1) || (0 && (!lt->inverse) && (lt->axis_id == 0)) || ((lt->convolutionStep) && ((lt->matrixConvolution > 1) || (lt->numKernels > 1)) && (stageAngle < 0)) || (lt->regAd > 1) || (0)) {
		//appendBarrier(lt, 1);
		if (!((lt->regAd > 1) && (stageSize * stageRadix == lt->dim / lt->stageRadix[lt->numStages - 1]) && (lt->stageRadix[lt->numStages - 1] == lt->regAd))) {
			char** tempID;
			tempID = (char**)malloc(sizeof(char*) * lt->threadRegister * lt->regAd);
			if (tempID) {
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					tempID[i] = (char*)malloc(sizeof(char) * 50);
					if (!tempID[i]) {
						for (uint64_t j = 0; j < i; j++) {
							free(tempID[j]);
							tempID[j] = 0;
						}
						free(tempID);
						tempID = 0;
						return FFT_ERROR_MALLOC_FAILED;
					}
				}
				res = AppendLineFromInput(lt, lt->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (lt->localSize[0] * threadStorage > lt->dim) {
					lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_x, threadStorage, lt->dim);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				for (uint64_t k = 0; k < lt->regAd; ++k) {
					uint64_t t = 0;
					if (k > 0) {
						res = appendBarrier(lt, 2);
						if (res != FFT_SUCCESS) return res;
						res = AppendLineFromInput(lt, lt->disableThreadsStart);
						if (res != FFT_SUCCESS) return res;
						if (lt->localSize[0] * threadStorage > lt->dim) {
							lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_x, threadStorage, lt->dim);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}
					for (uint64_t j = 0; j < threadRegister / stageRadix; j++) {
						sprintf(tempNum, "%" PRIu64 "", j * logicalGroupSize);
						res = AddReal(lt, lt->stageInvocationID, lt->gl_LocalInvocationID_x, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = MovReal(lt, lt->blockInvocationID, lt->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						sprintf(tempNum, "%" PRIu64 "", stageSize);
						res = ModReal(lt, lt->stageInvocationID, lt->stageInvocationID, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = SubReal(lt, lt->blockInvocationID, lt->blockInvocationID, lt->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						sprintf(tempNum, "%" PRIu64 "", stageRadix);
						res = MulReal(lt, lt->inoutID, lt->blockInvocationID, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = AddReal(lt, lt->inoutID, lt->inoutID, lt->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						

							for (uint64_t i = 0; i < stageRadix; i++) {
								uint64_t id = j + k * threadRegister / stageRadix + i * threadStorage / stageRadix;
								id = (id / threadRegister) * lt->threadRegister + id % threadRegister;
								sprintf(tempID[t + k * lt->threadRegister], "%s", lt->regIDs[id]);
								t++;
								sprintf(tempNum, "%" PRIu64 "", i * stageSize);
								res = AddReal(lt, lt->sdataID, lt->inoutID, tempNum);
								if (res != FFT_SUCCESS) return res;
								if ((stageSize <= lt->numSharedBanks / 2) && (lt->dim > lt->numSharedBanks / 2) && (lt->conflictStride != lt->dim / lt->regAd) && ((lt->dim & (lt->dim - 1)) == 0) && (stageSize * stageRadix != lt->dim)) {
									if (lt->conflictStages == 0) {
										lt->conflictStages = 1;
										lt->tempLen = sprintf(lt->tempStr, "\
%s = %" PRIu64 ";", lt->sharedStride, lt->conflictStride);
										res = AppendLine(lt);
										if (res != FFT_SUCCESS) return res;
									}
									lt->tempLen = sprintf(lt->tempStr, "\
	%s = (%s / %" PRIu64 ") * %" PRIu64 " + %s %% %" PRIu64 ";", lt->sdataID, lt->sdataID, lt->numSharedBanks / 2, lt->numSharedBanks / 2 + 1, lt->sdataID, lt->numSharedBanks / 2);
									res = AppendLine(lt);
									if (res != FFT_SUCCESS) return res;

								}
								else {
									if (lt->conflictStages == 1) {
										lt->conflictStages = 0;
										lt->tempLen = sprintf(lt->tempStr, "\
	%s = %" PRIu64 ";", lt->sharedStride, lt->conflictShared);
										res = AppendLine(lt);
										if (res != FFT_SUCCESS) return res;
									}
								}
								if (lt->localSize[1] > 1) {
									res = MulReal(lt, lt->combinedID, lt->gl_LocalInvocationID_y, lt->sharedStride);
									if (res != FFT_SUCCESS) return res;
									res = AddReal(lt, lt->sdataID, lt->sdataID, lt->combinedID);
									if (res != FFT_SUCCESS) return res;
								}
								if (strcmp(stageNormalization, "")) {
									res = DivComplexNumber(lt, lt->regIDs[id], lt->regIDs[id], stageNormalization);
									if (res != FFT_SUCCESS) return res;
								}
								res = SharedStore(lt, lt->sdataID, lt->regIDs[id]);
								if (res != FFT_SUCCESS) return res;

							}
					}
					for (uint64_t j = threadRegister; j < lt->threadRegister; j++) {
						sprintf(tempID[t + k * lt->threadRegister], "%s", lt->regIDs[t + k * lt->threadRegister]);
						t++;
					}
					t = 0;
					if (lt->regAd > 1) {
						if (lt->localSize[0] * threadStorage > lt->dim)
						{
							lt->tempLen = sprintf(lt->tempStr, "	}\n");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(lt, lt->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
						res = appendBarrier(lt, 2);
						if (res != FFT_SUCCESS) return res;

						res = AppendLineFromInput(lt, lt->disableThreadsStart);
						if (res != FFT_SUCCESS) return res;
						if (lt->localSize[0] * threadStorageNext > lt->dim) {
							lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_x, threadStorageNext, lt->dim);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						for (uint64_t j = 0; j < threadRegisterNext / stageRadixNext; j++) {
							for (uint64_t i = 0; i < stageRadixNext; i++) {
								uint64_t id = j + k * threadRegisterNext / stageRadixNext + i * threadStorageNext / stageRadixNext;
								id = (id / threadRegisterNext) * lt->threadRegister + id % threadRegisterNext;
								//resID[t + k * lt->threadRegister] = lt->regIDs[id];
								sprintf(tempNum, "%" PRIu64 "", t * logicalGroupSizeNext);
								res = AddReal(lt, lt->sdataID, lt->gl_LocalInvocationID_x, tempNum);
								if (res != FFT_SUCCESS) return res;
								if (lt->localSize[1] > 1) {
									res = MulReal(lt, lt->combinedID, lt->gl_LocalInvocationID_y, lt->sharedStride);
									if (res != FFT_SUCCESS) return res;
									res = AddReal(lt, lt->sdataID, lt->sdataID, lt->combinedID);
									if (res != FFT_SUCCESS) return res;
								}
								res = SharedLoad(lt, tempID[t + k * lt->threadRegister], lt->sdataID);
								if (res != FFT_SUCCESS) return res;
								
								t++;
							}

						}
						if (lt->localSize[0] * threadStorageNext > lt->dim)
						{
							lt->tempLen = sprintf(lt->tempStr, "	}\n");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(lt, lt->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						if (lt->localSize[0] * threadStorage > lt->dim)
						{
							lt->tempLen = sprintf(lt->tempStr, "	}\n");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(lt, lt->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
					}
				}
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					//printf("0 - %s\n", resID[i]);
					sprintf(lt->regIDs[i], "%s", tempID[i]);
					//sprintf(resID[i], "%s", tempID[i]);
					//printf("1 - %s\n", resID[i]);
				}
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					free(tempID[i]);
					tempID[i] = 0;
				}
				free(tempID);
				tempID = 0;
			}
			else
				return FFT_ERROR_MALLOC_FAILED;
		}
		else {
			char** tempID;
			tempID = (char**)malloc(sizeof(char*) * lt->threadRegister * lt->regAd);
			if (tempID) {
				//resID = (char**)malloc(sizeof(char*) * lt->threadRegister * lt->regAd);
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					tempID[i] = (char*)malloc(sizeof(char) * 50);
					if (!tempID[i]) {
						for (uint64_t j = 0; j < i; j++) {
							free(tempID[j]);
							tempID[j] = 0;
						}
						free(tempID);
						tempID = 0;
						return FFT_ERROR_MALLOC_FAILED;
					}
				}
				for (uint64_t k = 0; k < lt->regAd; ++k) {
					for (uint64_t j = 0; j < threadRegister / stageRadix; j++) {
						for (uint64_t i = 0; i < stageRadix; i++) {
							uint64_t id = j + k * threadRegister / stageRadix + i * threadStorage / stageRadix;
							id = (id / threadRegister) * lt->threadRegister + id % threadRegister;
							sprintf(tempID[j + i * threadRegister / stageRadix + k * lt->threadRegister], "%s", lt->regIDs[id]);
						}
					}
					for (uint64_t j = threadRegister; j < lt->threadRegister; j++) {
						sprintf(tempID[j + k * lt->threadRegister], "%s", lt->regIDs[j + k * lt->threadRegister]);
					}
				}
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					sprintf(lt->regIDs[i], "%s", tempID[i]);
				}
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					free(tempID[i]);
					tempID[i] = 0;
				}
				free(tempID);
				tempID = 0;
			}
			else
				return FFT_ERROR_MALLOC_FAILED;
		}
	}
	else {
		res = AppendLineFromInput(lt, lt->disableThreadsStart);
		if (res != FFT_SUCCESS) return res;
		if (lt->localSize[0] * threadStorage > lt->dim) {
			lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_x, threadStorage, lt->dim);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
		}
		if (((lt->actualInverse) && (lt->normalize)) || ((lt->convolutionStep) && (stageAngle > 0))) {
			for (uint64_t i = 0; i < threadStorage; i++) {
				res = DivComplexNumber(lt, lt->regIDs[(i / threadRegister) * lt->threadRegister + i % threadRegister], lt->regIDs[(i / threadRegister) * lt->threadRegister + i % threadRegister], stageNormalization);
				if (res != FFT_SUCCESS) return res;
				/*lt->tempLen = sprintf(lt->tempStr, "\
	temp%s = temp%s%s;\n", lt->regIDs[(i / threadRegister) * lt->threadRegister + i % threadRegister], lt->regIDs[(i / threadRegister) * lt->threadRegister + i % threadRegister], stageNormalization);*/
			}
		}
		if (lt->localSize[0] * threadStorage > lt->dim)
		{
			lt->tempLen = sprintf(lt->tempStr, "	}\n");
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
		}
		res = AppendLineFromInput(lt, lt->disableThreadsEnd);
		if (res != FFT_SUCCESS) return res;
	}
	return res;
}


FFTResult appendRadixShuffleStrided(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");


	char stageNormalization[10] = "";
	char tempNum[50] = "";

	uint64_t threadStorage = lt->threadRadixRegister[stageRadix] * lt->regAd;// (lt->threadRegister % stageRadix == 0) ? lt->threadRegister * lt->regAd : lt->threadRegisterMin * lt->regAd;
	uint64_t threadStorageNext = lt->threadRadixRegister[stageRadixNext] * lt->regAd;//(lt->threadRegister % stageRadixNext == 0) ? lt->threadRegister * lt->regAd : lt->threadRegisterMin * lt->regAd;
	uint64_t threadRegister = lt->threadRadixRegister[stageRadix];//(lt->threadRegister % stageRadix == 0) ? lt->threadRegister : lt->threadRegisterMin;
	uint64_t threadRegisterNext = lt->threadRadixRegister[stageRadixNext];//(lt->threadRegister % stageRadixNext == 0) ? lt->threadRegister : lt->threadRegisterMin;

	uint64_t logicalGroupSize = lt->dim / threadStorage;
	uint64_t logicalGroupSizeNext = lt->dim / threadStorageNext;
	if (((lt->actualInverse) && (lt->normalize)) || ((lt->convolutionStep) && (stageAngle > 0))) {
		if ((stageSize == 1) && (0)) {
			if (0 == 4)
				sprintf(stageNormalization, "%" PRIu64 "", stageRadix * 4);
			else
				sprintf(stageNormalization, "%" PRIu64 "", stageRadix * 2);
		}
		else
			sprintf(stageNormalization, "%" PRIu64 "", stageRadix);
	}
	if (((lt->axis_id == 0) && (lt->axis_upload_id == 0)) || (lt->localSize[1] * threadStorage > lt->dim) || (stageSize < lt->dim / stageRadix) || ((lt->convolutionStep) && ((lt->matrixConvolution > 1) || (lt->numKernels > 1)) && (stageAngle < 0)) || (0))
	{
		res = appendBarrier(lt, 2);
		if (res != FFT_SUCCESS) return res;
	}
	if (stageSize == lt->dim / stageRadix) {
		lt->tempLen = sprintf(lt->tempStr, "		%s = %" PRIu64 ";\n", lt->sharedStride, lt->conflictShared);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
	}
	if (((lt->axis_id == 0) && (lt->axis_upload_id == 0)) || (lt->localSize[1] * threadStorage > lt->dim) || (stageSize < lt->dim / stageRadix) || ((lt->convolutionStep) && ((lt->matrixConvolution > 1) || (lt->numKernels > 1)) && (stageAngle < 0)) || (0)) {
		//appendBarrier(lt, 2);
		if (!((lt->regAd > 1) && (stageSize * stageRadix == lt->dim / lt->stageRadix[lt->numStages - 1]) && (lt->stageRadix[lt->numStages - 1] == lt->regAd))) {
			char** tempID;
			tempID = (char**)malloc(sizeof(char*) * lt->threadRegister * lt->regAd);
			if (tempID) {
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					tempID[i] = (char*)malloc(sizeof(char) * 50);
					if (!tempID[i]) {
						for (uint64_t j = 0; j < i; j++) {
							free(tempID[j]);
							tempID[j] = 0;
						}
						free(tempID);
						tempID = 0;
						return FFT_ERROR_MALLOC_FAILED;
					}
				}
				res = AppendLineFromInput(lt, lt->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (lt->localSize[1] * threadStorage > lt->dim) {
					lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_LocalInvocationID_y, threadStorage, lt->dim);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				for (uint64_t k = 0; k < lt->regAd; ++k) {
					uint64_t t = 0;
					if (k > 0) {
						res = appendBarrier(lt, 2);
						if (res != FFT_SUCCESS) return res;
						res = AppendLineFromInput(lt, lt->disableThreadsStart);
						if (res != FFT_SUCCESS) return res;
						if (lt->localSize[1] * threadStorage > lt->dim) {
							lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_LocalInvocationID_y, threadStorage, lt->dim);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}
					for (uint64_t j = 0; j < threadRegister / stageRadix; j++) {
						sprintf(tempNum, "%" PRIu64 "", j * logicalGroupSize);
						res = AddReal(lt, lt->stageInvocationID, lt->gl_LocalInvocationID_y, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = MovReal(lt, lt->blockInvocationID, lt->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						sprintf(tempNum, "%" PRIu64 "", stageSize);
						res = ModReal(lt, lt->stageInvocationID, lt->stageInvocationID, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = SubReal(lt, lt->blockInvocationID, lt->blockInvocationID, lt->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						sprintf(tempNum, "%" PRIu64 "", stageRadix);
						res = MulReal(lt, lt->inoutID, lt->blockInvocationID, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = AddReal(lt, lt->inoutID, lt->inoutID, lt->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						
						for (uint64_t i = 0; i < stageRadix; i++) {
							uint64_t id = j + k * threadRegister / stageRadix + i * threadStorage / stageRadix;
							id = (id / threadRegister) * lt->threadRegister + id % threadRegister;
							sprintf(tempID[t + k * lt->threadRegister], "%s", lt->regIDs[id]);
							t++;
							sprintf(tempNum, "%" PRIu64 "", i * stageSize);
							res = AddReal(lt, lt->sdataID, lt->inoutID, tempNum);
							if (res != FFT_SUCCESS) return res;
							res = MulReal(lt, lt->sdataID, lt->sharedStride, lt->sdataID);
							if (res != FFT_SUCCESS) return res;
							res = AddReal(lt, lt->sdataID, lt->sdataID, lt->gl_LocalInvocationID_x);
							if (res != FFT_SUCCESS) return res;
							//sprintf(lt->sdataID, "sharedStride * gl_LocalInvocationID.y + inoutID + %" PRIu64 "", i * stageSize);
							if (strcmp(stageNormalization, "")) {
								res = DivComplexNumber(lt, lt->regIDs[id], lt->regIDs[id], stageNormalization);
								if (res != FFT_SUCCESS) return res;
							}
							res = SharedStore(lt, lt->sdataID, lt->regIDs[id]);
							if (res != FFT_SUCCESS) return res;
						
						}
					}
					for (uint64_t j = threadRegister; j < lt->threadRegister; j++) {
						sprintf(tempID[t + k * lt->threadRegister], "%s", lt->regIDs[t + k * lt->threadRegister]);
						t++;
					}
					t = 0;
					if (lt->regAd > 1) {
						if (lt->localSize[1] * threadStorage > lt->dim)
						{
							lt->tempLen = sprintf(lt->tempStr, "	}\n");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(lt, lt->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
						res = appendBarrier(lt, 2);
						if (res != FFT_SUCCESS) return res;
						res = AppendLineFromInput(lt, lt->disableThreadsStart);
						if (res != FFT_SUCCESS) return res;
						if (lt->localSize[1] * threadStorageNext > lt->dim) {
							lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_LocalInvocationID_y, threadStorageNext, lt->dim);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						for (uint64_t j = 0; j < threadRegisterNext / stageRadixNext; j++) {
							for (uint64_t i = 0; i < stageRadixNext; i++) {
								uint64_t id = j + k * threadRegisterNext / stageRadixNext + i * threadRegisterNext / stageRadixNext;
								id = (id / threadRegisterNext) * lt->threadRegister + id % threadRegisterNext;
								sprintf(tempNum, "%" PRIu64 "", t * logicalGroupSizeNext);
								res = AddReal(lt, lt->sdataID, lt->gl_LocalInvocationID_y, tempNum);
								if (res != FFT_SUCCESS) return res;
								res = MulReal(lt, lt->sdataID, lt->sharedStride, lt->sdataID);
								if (res != FFT_SUCCESS) return res;
								res = AddReal(lt, lt->sdataID, lt->sdataID, lt->gl_LocalInvocationID_x);
								if (res != FFT_SUCCESS) return res;
								res = SharedLoad(lt, tempID[t + k * lt->threadRegister], lt->sdataID);
								if (res != FFT_SUCCESS) return res;
								
								t++;
							}
						}
						if (lt->localSize[1] * threadStorageNext > lt->dim)
						{
							lt->tempLen = sprintf(lt->tempStr, "	}\n");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(lt, lt->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						if (lt->localSize[1] * threadStorage > lt->dim)
						{
							lt->tempLen = sprintf(lt->tempStr, "	}\n");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(lt, lt->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
					}
				}
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					sprintf(lt->regIDs[i], "%s", tempID[i]);
				}
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					free(tempID[i]);
					tempID[i] = 0;
				}
				free(tempID);
				tempID = 0;
			}
			else
				return FFT_ERROR_MALLOC_FAILED;
		}
		else {
			char** tempID;
			tempID = (char**)malloc(sizeof(char*) * lt->threadRegister * lt->regAd);
			if (tempID) {
				//resID = (char**)malloc(sizeof(char*) * lt->threadRegister * lt->regAd);
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					tempID[i] = (char*)malloc(sizeof(char) * 50);
					if (!tempID[i]) {
						for (uint64_t j = 0; j < i; j++) {
							free(tempID[j]);
							tempID[j] = 0;
						}
						free(tempID);
						tempID = 0;
						return FFT_ERROR_MALLOC_FAILED;
					}
				}
				for (uint64_t k = 0; k < lt->regAd; ++k) {
					for (uint64_t j = 0; j < threadRegister / stageRadix; j++) {
						for (uint64_t i = 0; i < stageRadix; i++) {
							uint64_t id = j + k * threadRegister / stageRadix + i * threadStorage / stageRadix;
							id = (id / threadRegister) * lt->threadRegister + id % threadRegister;
							sprintf(tempID[j + i * threadRegister / stageRadix + k * lt->threadRegister], "%s", lt->regIDs[id]);
						}
					}
					for (uint64_t j = threadRegister; j < lt->threadRegister; j++) {
						sprintf(tempID[j + k * lt->threadRegister], "%s", lt->regIDs[j + k * lt->threadRegister]);
					}
				}
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					sprintf(lt->regIDs[i], "%s", tempID[i]);
				}
				for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
					free(tempID[i]);
					tempID[i] = 0;
				}
				free(tempID);
				tempID = 0;
			}
			else
				return FFT_ERROR_MALLOC_FAILED;
		}
	}
	else {
		res = AppendLineFromInput(lt, lt->disableThreadsStart);
		if (res != FFT_SUCCESS) return res;
		if (lt->localSize[1] * threadStorage > lt->dim) {
			lt->tempLen = sprintf(lt->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", lt->gl_LocalInvocationID_y, threadStorage, lt->dim);
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
		}
		if (((lt->actualInverse) && (lt->normalize)) || ((lt->convolutionStep) && (stageAngle > 0))) {
			for (uint64_t i = 0; i < threadRegister; i++) {
				res = DivComplexNumber(lt, lt->regIDs[(i / threadRegister) * lt->threadRegister + i % threadRegister], lt->regIDs[(i / threadRegister) * lt->threadRegister + i % threadRegister], stageNormalization);
				if (res != FFT_SUCCESS) return res;
			}
		}
		if (lt->localSize[1] * threadRegister > lt->dim)
		{
			lt->tempLen = sprintf(lt->tempStr, "	}\n");
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
		}
		res = AppendLineFromInput(lt, lt->disableThreadsEnd);
		if (res != FFT_SUCCESS) return res;
	}
	return res;
}



FFTResult appendRadixShuffle(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext, uint64_t shuffleType) {
	FFTResult res = FFT_SUCCESS;
	switch (shuffleType) {
	case 0: case 5: case 6: case 120: case 130: case 140: case 142: {
		res = appendRadixShuffleNonStrided(lt, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix, stageRadixNext);
		if (res != FFT_SUCCESS) return res;
		//appendBarrier(lt, 1);
		break;
	}
	case 1: case 2: case 121: case 131: case 141: case 143: {
		res = appendRadixShuffleStrided(lt, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix, stageRadixNext);
		if (res != FFT_SUCCESS) return res;
		//appendBarrier(lt, 1);
		break;
	}
	}
	return res;
}



FFTResult appendReorder4StepWrite(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t reorderType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char LFending[4] = "";

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	char cosDef[20] = "__cosf";
	char sinDef[20] = "__sinf";
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");


	uint64_t threadRegister = lt->threadRadixRegister[lt->stageRadix[lt->numStages - 1]];// (lt->threadRegister % lt->stageRadix[lt->numStages - 1] == 0) ? lt->threadRegister : lt->threadRegisterMin;
	switch (reorderType) {
	case 1: {//grouped_c2c
		char shiftX[500] = "";
		if (lt->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", lt->gl_WorkGroupSize_x);
		if ((lt->stageStartSize > 1) && (!((lt->stageStartSize > 1) && (!lt->reorderFourStep) && (lt->inverse)))) {
			if (lt->localSize[1] * lt->stageRadix[lt->numStages - 1] * (lt->threadRadixRegister[lt->stageRadix[lt->numStages - 1]] / lt->stageRadix[lt->numStages - 1]) > lt->dim) {
				res = appendBarrier(lt, 1);
				if (res != FFT_SUCCESS) return res;
				lt->writeFromRegisters = 0;
			}
			else
				lt->writeFromRegisters = 1;
			res = AppendLineFromInput(lt, lt->disableThreadsStart);
			if (res != FFT_SUCCESS) return res;
			for (uint64_t i = 0; i < lt->dim / lt->localSize[1]; i++) {
				uint64_t id = (i / threadRegister) * lt->threadRegister + i % threadRegister;
				if (lt->LUT) {
					lt->tempLen = sprintf(lt->tempStr, "		mult = twiddleLUT[%" PRIu64 "+(((%s%s)/%" PRIu64 ") %% (%" PRIu64 "))+%" PRIu64 "*(%s+%" PRIu64 ")];\n", lt->maxStageSumLUT, lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x, lt->stageStartSize, lt->stageStartSize, lt->gl_LocalInvocationID_y, i * lt->localSize[1]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (!lt->inverse) {
						lt->tempLen = sprintf(lt->tempStr, "	mult.y = -mult.y;\n");
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					lt->tempLen = sprintf(lt->tempStr, "		angle = 2 * loc_PI * ((((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")) * (%s + %" PRIu64 ")) / %f%s;\n", lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x, lt->stageStartSize, lt->gl_LocalInvocationID_y, i * lt->localSize[1], (double)(lt->stageStartSize * lt->dim), LFending);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (lt->inverse) {
						if (!strcmp(floatType, "double")) {
							lt->tempLen = sprintf(lt->tempStr, "		mult = sincos_20(angle);\n");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}
					else {
						if (!strcmp(floatType, "double")) {
							lt->tempLen = sprintf(lt->tempStr, "		mult = sincos_20(-angle);\n");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}
				}
				if (lt->writeFromRegisters) {
					lt->tempLen = sprintf(lt->tempStr, "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", lt->regIDs[id], lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", lt->regIDs[id], lt->regIDs[id], lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		%s.x = w.x;\n", lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					lt->tempLen = sprintf(lt->tempStr, "\
		%s = %s*(%" PRIu64 "+%s) + %s;\n", lt->inoutID, lt->sharedStride, i * lt->localSize[1], lt->gl_LocalInvocationID_y, lt->gl_LocalInvocationID_x);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					lt->tempLen = sprintf(lt->tempStr, "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", lt->inoutID, lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					lt->tempLen = sprintf(lt->tempStr, "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", lt->inoutID, lt->inoutID, lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		sdata[%s].x = w.x;\n", lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
			}
			res = AppendLineFromInput(lt, lt->disableThreadsEnd);
			if (res != FFT_SUCCESS) return res;
		}
		break;
	}
	case 2: {//single_c2c_strided
		char shiftX[500] = "";
		if (lt->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", lt->gl_WorkGroupSize_x);
		if (!((!lt->reorderFourStep) && (lt->inverse))) {
			if (lt->localSize[1] * lt->stageRadix[lt->numStages - 1] * (lt->threadRadixRegister[lt->stageRadix[lt->numStages - 1]] / lt->stageRadix[lt->numStages - 1]) > lt->dim) {
				res = appendBarrier(lt, 1);
				if (res != FFT_SUCCESS) return res;
				lt->writeFromRegisters = 0;
			}
			else
				lt->writeFromRegisters = 1;
			res = AppendLineFromInput(lt, lt->disableThreadsStart);
			if (res != FFT_SUCCESS) return res;
			for (uint64_t i = 0; i < lt->dim / lt->localSize[1]; i++) {
				uint64_t id = (i / threadRegister) * lt->threadRegister + i % threadRegister;
				if (lt->LUT) {
					lt->tempLen = sprintf(lt->tempStr, "		mult = twiddleLUT[%" PRIu64 " + ((%s%s) %% (%" PRIu64 ")) + (%s + %" PRIu64 ") * %" PRIu64 "];\n", lt->maxStageSumLUT, lt->gl_GlobalInvocationID_x, shiftX, lt->stageStartSize, lt->gl_LocalInvocationID_y, i * lt->localSize[1], lt->stageStartSize);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (!lt->inverse) {
						lt->tempLen = sprintf(lt->tempStr, "	mult.y = -mult.y;\n");
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					lt->tempLen = sprintf(lt->tempStr, "		angle = 2 * loc_PI * ((((%s%s) %% (%" PRIu64 ")) * (%s + %" PRIu64 ")) / %f%s);\n", lt->gl_GlobalInvocationID_x, shiftX, lt->stageStartSize, lt->gl_LocalInvocationID_y, i * lt->localSize[1], (double)(lt->stageStartSize * lt->dim), LFending);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (lt->inverse) {
						if (!strcmp(floatType, "double")) {
							lt->tempLen = sprintf(lt->tempStr, "		mult = sincos_20(angle);\n");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}
					else {
						if (!strcmp(floatType, "double")) {
							lt->tempLen = sprintf(lt->tempStr, "		mult = sincos_20(-angle);\n");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}
				}
				if (lt->writeFromRegisters) {
					lt->tempLen = sprintf(lt->tempStr, "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", lt->regIDs[id], lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", lt->regIDs[id], lt->regIDs[id], lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		%s.x = w.x;\n", lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					lt->tempLen = sprintf(lt->tempStr, "\
		%s = %s*(%" PRIu64 "+%s) + %s;\n", lt->inoutID, lt->sharedStride, i * lt->localSize[1], lt->gl_LocalInvocationID_y, lt->gl_LocalInvocationID_x);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					lt->tempLen = sprintf(lt->tempStr, "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", lt->inoutID, lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					lt->tempLen = sprintf(lt->tempStr, "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", lt->inoutID, lt->inoutID, lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		sdata[%s].x = w.x;\n", lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
			}
			res = AppendLineFromInput(lt, lt->disableThreadsEnd);
			if (res != FFT_SUCCESS) return res;
		}
		break;
	}
	}
	return res;
}


FFTResult indexInputFFT(FFTLayout* lt, const char* uintType, uint64_t inputType, const char* index_x, const char* index_y, const char* coordinate, const char* batchID) {
	FFTResult res = FFT_SUCCESS;
	switch (inputType) {
	case 0: case 2: case 3: case 4:case 5: case 6: case 120: case 130: case 140: case 142: {
		char inputOffset[30] = "";
		if (lt->inputOffset > 0)
			sprintf(inputOffset, "%" PRIu64 " + ", lt->inputOffset);
		char shiftX[500] = "";
		if (lt->inputStride[0] == 1)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, lt->inputStride[0]);
		char shiftY[500] = "";
		uint64_t mult = (lt->mergeSequencesR2C) ? 2 : 1;
		if (lt->size[1] > 1) {
			if (lt->dim == lt->fft_dim_full) {
				if (lt->axisSwapped) {
					if (lt->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", lt->gl_WorkGroupID_y, mult * lt->localSize[0] * lt->inputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", lt->gl_WorkGroupID_y, mult * lt->localSize[0] * lt->inputStride[1]);
				}
				else {
					if (lt->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", lt->gl_WorkGroupID_y, mult * lt->localSize[1] * lt->inputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", lt->gl_WorkGroupID_y, mult * lt->localSize[1] * lt->inputStride[1]);
				}
			}
			else {
				if (lt->performWorkGroupShift[1])
					sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", lt->gl_WorkGroupID_y, lt->inputStride[1]);
				else
					sprintf(shiftY, " + %s * %" PRIu64 "", lt->gl_WorkGroupID_y, lt->inputStride[1]);
			}
		}
		char shiftZ[500] = "";
		if (lt->size[2] > 1) {
			if (lt->performWorkGroupShift[2])
				sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", lt->gl_GlobalInvocationID_z, lt->gl_WorkGroupSize_z, lt->inputStride[2]);
			else
				sprintf(shiftZ, " + %s * %" PRIu64 "", lt->gl_GlobalInvocationID_z, lt->inputStride[2]);
		}
		char shiftCoordinate[100] = "";
		if ((lt->matrixConvolution > 1) && (lt->convolutionStep)) {
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, lt->inputStride[3]);
		}
		char shiftBatch[100] = "";
		if ((lt->numBatches > 1) || (lt->numKernels > 1)) {
			if (lt->convolutionStep) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, lt->inputStride[4]);
			}
			else
				sprintf(shiftBatch, " + consts.batchID * %" PRIu64 "", lt->inputStride[4]);
		}
		lt->tempLen = sprintf(lt->tempStr, "%s%s%s%s%s%s", inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 1: case 121: case 131: case 141: case 143: {
		char inputOffset[30] = "";
		if (lt->inputOffset > 0)
			sprintf(inputOffset, "%" PRIu64 " + ", lt->inputOffset);
		char shiftX[500] = "";
		if (lt->inputStride[0] == 1)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, lt->inputStride[0]);

		char shiftY[500] = "";
		if (index_y)
			sprintf(shiftY, " + (%s) * %" PRIu64 "", index_y, lt->inputStride[1]);

		char shiftZ[500] = "";
		if (lt->size[2] > 1) {
			if (lt->performWorkGroupShift[2])
				sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", lt->gl_GlobalInvocationID_z, lt->gl_WorkGroupSize_z, lt->inputStride[2]);
			else
				sprintf(shiftZ, " + %s * %" PRIu64 "", lt->gl_GlobalInvocationID_z, lt->inputStride[2]);
		}
		char shiftCoordinate[100] = "";
		if ((lt->matrixConvolution > 1) && (lt->convolutionStep)) {
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, lt->inputStride[3]);
		}
		char shiftBatch[100] = "";
		if ((lt->numBatches > 1) || (lt->numKernels > 1)) {
			if (lt->convolutionStep) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, lt->inputStride[4]);
			}
			else
				sprintf(shiftBatch, " + consts.batchID * %" PRIu64 "", lt->inputStride[4]);
		}
		lt->tempLen = sprintf(lt->tempStr, "%s%s%s%s%s%s", inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	}
	return res;
}

FFTResult indexOutputFFT(FFTLayout* lt, const char* uintType, uint64_t outputType, const char* index_x, const char* index_y, const char* coordinate, const char* batchID) {
	FFTResult res = FFT_SUCCESS;
	switch (outputType) {//single_c2c + single_c2c_strided
	case 0: case 2: case 3: case 4: case 5: case 6: case 120: case 130: case 140: case 142: {
		char outputOffset[30] = "";
		if (lt->outputOffset > 0)
			sprintf(outputOffset, "%" PRIu64 " + ", lt->outputOffset);
		char shiftX[500] = "";
		if (lt->dim == lt->fft_dim_full)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, lt->outputStride[0]);
		char shiftY[500] = "";
		uint64_t mult = (lt->mergeSequencesR2C) ? 2 : 1;
		if (lt->size[1] > 1) {
			if (lt->dim == lt->fft_dim_full) {
				if (lt->axisSwapped) {
					if (lt->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", lt->gl_WorkGroupID_y, mult * lt->localSize[0] * lt->outputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", lt->gl_WorkGroupID_y, mult * lt->localSize[0] * lt->outputStride[1]);
				}
				else {
					if (lt->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", lt->gl_WorkGroupID_y, mult * lt->localSize[1] * lt->outputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", lt->gl_WorkGroupID_y, mult * lt->localSize[1] * lt->outputStride[1]);
				}
			}
			else {
				if (lt->performWorkGroupShift[1])
					sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", lt->gl_WorkGroupID_y, lt->outputStride[1]);
				else
					sprintf(shiftY, " + %s * %" PRIu64 "", lt->gl_WorkGroupID_y, lt->outputStride[1]);
			}
		}
		char shiftZ[500] = "";
		if (lt->size[2] > 1) {
			if (lt->performWorkGroupShift[2])
				sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", lt->gl_GlobalInvocationID_z, lt->gl_WorkGroupSize_z, lt->outputStride[2]);
			else
				sprintf(shiftZ, " + %s * %" PRIu64 "", lt->gl_GlobalInvocationID_z, lt->outputStride[2]);
		}
		char shiftCoordinate[100] = "";
		if ((lt->matrixConvolution > 1) && (lt->convolutionStep)) {
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, lt->outputStride[3]);
		}
		char shiftBatch[100] = "";
		if ((lt->numBatches > 1) || (lt->numKernels > 1)) {
			if (lt->convolutionStep) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, lt->outputStride[4]);
			}
			else
				sprintf(shiftBatch, " + consts.batchID * %" PRIu64 "", lt->outputStride[4]);
		}
		lt->tempLen = sprintf(lt->tempStr, "%s%s%s%s%s%s", outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 1: case 121: case 131: case 141: case 143: {//grouped_c2c
		char outputOffset[30] = "";
		if (lt->outputOffset > 0)
			sprintf(outputOffset, "%" PRIu64 " + ", lt->outputOffset);
		char shiftX[500] = "";
		if (lt->dim == lt->fft_dim_full)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, lt->outputStride[0]);
		char shiftY[500] = "";
		if (index_y)
			sprintf(shiftY, " + (%s) * %" PRIu64 "", index_y, lt->outputStride[1]);
		char shiftZ[500] = "";
		if (lt->size[2] > 1) {
			if (lt->performWorkGroupShift[2])
				sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", lt->gl_GlobalInvocationID_z, lt->gl_WorkGroupSize_z, lt->outputStride[2]);
			else
				sprintf(shiftZ, " + %s * %" PRIu64 "", lt->gl_GlobalInvocationID_z, lt->outputStride[2]);
		}
		char shiftCoordinate[100] = "";
		if ((lt->matrixConvolution > 1) && (lt->convolutionStep)) {
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, lt->outputStride[3]);
		}
		char shiftBatch[100] = "";
		if ((lt->numBatches > 1) || (lt->numKernels > 1)) {
			if (lt->convolutionStep) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, lt->outputStride[4]);
			}
			else
				sprintf(shiftBatch, " + consts.batchID * %" PRIu64 "", lt->outputStride[4]);
		}
		lt->tempLen = sprintf(lt->tempStr, "%s%s%s%s%s%s", outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		break;

	}
	}
	return res;
}

FFTResult appendReorder4StepRead(FFTLayout* lt, const char* floatType, const char* uintType, uint64_t reorderType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char LFending[4] = "";

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	char cosDef[20] = "__cosf";
	char sinDef[20] = "__sinf";
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");


	uint64_t threadRegister = lt->threadRadixRegister[lt->stageRadix[0]];// (lt->threadRegister % lt->stageRadix[lt->numStages - 1] == 0) ? lt->threadRegister : lt->threadRegisterMin;
	switch (reorderType) {
	case 1: {//grouped_c2c
		char shiftX[500] = "";
		if (lt->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", lt->gl_WorkGroupSize_x);
		if ((lt->stageStartSize > 1) && (!lt->reorderFourStep) && (lt->inverse)) {
			if (lt->localSize[1] * lt->stageRadix[0] * (lt->threadRadixRegister[lt->stageRadix[0]] / lt->stageRadix[0]) > lt->dim) {
				res = appendBarrier(lt, 1);
				if (res != FFT_SUCCESS) return res;
				lt->readToRegisters = 0;
			}
			else
				lt->readToRegisters = 1;
			res = AppendLineFromInput(lt, lt->disableThreadsStart);
			if (res != FFT_SUCCESS) return res;
			for (uint64_t i = 0; i < lt->dim / lt->localSize[1]; i++) {
				uint64_t id = (i / threadRegister) * lt->threadRegister + i % threadRegister;
				if (lt->LUT) {
					lt->tempLen = sprintf(lt->tempStr, "		mult = twiddleLUT[%" PRIu64 "+(((%s%s)/%" PRIu64 ") %% (%" PRIu64 "))+%" PRIu64 "*(%s+%" PRIu64 ")];\n", lt->maxStageSumLUT, lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x, lt->stageStartSize, lt->stageStartSize, lt->gl_LocalInvocationID_y, i * lt->localSize[1]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (!lt->inverse) {
						lt->tempLen = sprintf(lt->tempStr, "	mult.y = -mult.y;\n");
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					lt->tempLen = sprintf(lt->tempStr, "		angle = 2 * loc_PI * ((((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")) * (%s + %" PRIu64 ")) / %f%s;\n", lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x, lt->stageStartSize, lt->gl_LocalInvocationID_y, i * lt->localSize[1], (double)(lt->stageStartSize * lt->dim), LFending);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (!strcmp(floatType, "double")) {
						lt->tempLen = sprintf(lt->tempStr, "		mult = sincos_20(angle);\n");
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				if (lt->readToRegisters) {
					lt->tempLen = sprintf(lt->tempStr, "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", lt->regIDs[id], lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", lt->regIDs[id], lt->regIDs[id], lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		%s.x = w.x;\n", lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					lt->tempLen = sprintf(lt->tempStr, "\
		%s = %s*(%" PRIu64 "+%s) + %s;\n", lt->inoutID, lt->sharedStride, i * lt->localSize[1], lt->gl_LocalInvocationID_y, lt->gl_LocalInvocationID_x);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					lt->tempLen = sprintf(lt->tempStr, "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", lt->inoutID, lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					lt->tempLen = sprintf(lt->tempStr, "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", lt->inoutID, lt->inoutID, lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		sdata[%s].x = w.x;\n", lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
			}
			res = AppendLineFromInput(lt, lt->disableThreadsEnd);
			if (res != FFT_SUCCESS) return res;
		}

		break;
	}
	case 2: {//single_c2c_strided
		char shiftX[500] = "";
		if (lt->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", lt->gl_WorkGroupSize_x);
		if ((!lt->reorderFourStep) && (lt->inverse)) {
			if (lt->localSize[1] * lt->stageRadix[0] * (lt->threadRadixRegister[lt->stageRadix[0]] / lt->stageRadix[0]) > lt->dim) {
				res = appendBarrier(lt, 1);
				lt->readToRegisters = 0;
			}
			else
				lt->readToRegisters = 1;
			res = AppendLineFromInput(lt, lt->disableThreadsStart);
			if (res != FFT_SUCCESS) return res;
			for (uint64_t i = 0; i < lt->dim / lt->localSize[1]; i++) {
				uint64_t id = (i / threadRegister) * lt->threadRegister + i % threadRegister;
				if (lt->LUT) {
					lt->tempLen = sprintf(lt->tempStr, "		mult = twiddleLUT[%" PRIu64 " + ((%s%s) %% (%" PRIu64 ")) + (%s + %" PRIu64 ") * %" PRIu64 "];\n", lt->maxStageSumLUT, lt->gl_GlobalInvocationID_x, shiftX, lt->stageStartSize, lt->gl_LocalInvocationID_y, i * lt->localSize[1], lt->stageStartSize);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (!lt->inverse) {
						lt->tempLen = sprintf(lt->tempStr, "	mult.y = -mult.y;\n");
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					lt->tempLen = sprintf(lt->tempStr, "		angle = 2 * loc_PI * ((((%s%s) %% (%" PRIu64 ")) * (%s + %" PRIu64 ")) / %f%s);\n", lt->gl_GlobalInvocationID_x, shiftX, lt->stageStartSize, lt->gl_LocalInvocationID_y, i * lt->localSize[1], (double)(lt->stageStartSize * lt->dim), LFending);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					if (!strcmp(floatType, "double")) {
						lt->tempLen = sprintf(lt->tempStr, "		mult = sincos_20(angle);\n");
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
				if (lt->readToRegisters) {
					lt->tempLen = sprintf(lt->tempStr, "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", lt->regIDs[id], lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", lt->regIDs[id], lt->regIDs[id], lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		%s.x = w.x;\n", lt->regIDs[id]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					lt->tempLen = sprintf(lt->tempStr, "\
		%s = %s*(%" PRIu64 "+%s) + %s;\n", lt->inoutID, lt->sharedStride, i * lt->localSize[1], lt->gl_LocalInvocationID_y, lt->gl_LocalInvocationID_x);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					lt->tempLen = sprintf(lt->tempStr, "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", lt->inoutID, lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					lt->tempLen = sprintf(lt->tempStr, "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", lt->inoutID, lt->inoutID, lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, "\
		sdata[%s].x = w.x;\n", lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
			}
			res = AppendLineFromInput(lt, lt->disableThreadsEnd);
			if (res != FFT_SUCCESS) return res;
		}
		break;
	}
	}
	return res;
};

FFTResult appendReadDataFFT(FFTLayout* lt, const char* floatType, const char* floatTypeMemory, const char* uintType, uint64_t readType) {
	FFTResult res = FFT_SUCCESS;
	double double_PI = 3.1415926535897932384626433832795;
	char vecType[30];
	char inputsStruct[20] = "";
	char LFending[4] = "";

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");
	sprintf(inputsStruct, "inputs");
	char cosDef[20] = "__cosf";
	char sinDef[20] = "__sinf";

	char convTypeLeft[20] = "";
	char convTypeRight[20] = "";

	if ((!strcmp(floatType, "double")) && (strcmp(floatTypeMemory, "double"))) {
		if ((readType == 5) || (readType == 120) || (readType == 121) || (readType == 130) || (readType == 131) || (readType == 140) || (readType == 141) || (readType == 142) || (readType == 143)) {
			sprintf(convTypeLeft, "(double)");
		}
		else {
			sprintf(convTypeLeft, "conv_double2(");
			sprintf(convTypeRight, ")");
		}
	}
	char index_x[2000] = "";
	char index_y[2000] = "";
	char requestCoordinate[100] = "";

	char requestBatch[100] = "";

	switch (readType) {
	case 0:
	{
		char shiftX[500] = "";
		if (lt->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX ");
		char shiftY[500] = "";
		if (lt->axisSwapped) {
			if (lt->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*%s ", lt->gl_WorkGroupSize_x);
		}
		else {
			if (lt->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*%s ", lt->gl_WorkGroupSize_y);
		}
		char shiftY2[100] = "";
		if (lt->performWorkGroupShift[1])
			sprintf(shiftY, " + consts.workGroupShiftY ");
		if (lt->dim < lt->fft_dim_full) {
			if (lt->axisSwapped) {
				lt->tempLen = sprintf(lt->tempStr, "		%s numActiveThreads = ((%s/%" PRIu64 ")==%" PRIu64 ") ? %" PRIu64 " : %" PRIu64 ";\n", uintType, lt->gl_WorkGroupID_x, lt->firstStageStartSize / lt->dim, ((uint64_t)floor(lt->fft_dim_full / ((double)lt->localSize[0] * lt->dim))) / (lt->firstStageStartSize / lt->dim), (lt->fft_dim_full - (lt->firstStageStartSize / lt->dim) * ((((uint64_t)floor(lt->fft_dim_full / ((double)lt->localSize[0] * lt->dim))) / (lt->firstStageStartSize / lt->dim)) * lt->localSize[0] * lt->dim)) / lt->threadRegisterMin / (lt->firstStageStartSize / lt->dim), lt->localSize[0] * lt->localSize[1]);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
				sprintf(lt->disableThreadsStart, "		if(%s * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ") < %" PRIu64 ") {\n", lt->gl_LocalInvocationID_x, lt->firstStageStartSize, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->dim, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->localSize[0] * lt->firstStageStartSize, lt->fft_dim_full);
				lt->tempLen = sprintf(lt->tempStr, "		if((%s+%" PRIu64 "*%s)< numActiveThreads) {\n", lt->gl_LocalInvocationID_x, lt->localSize[0], lt->gl_LocalInvocationID_y);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
				sprintf(lt->disableThreadsEnd, "}");
			}
			else {
				sprintf(lt->disableThreadsStart, "		if(%s * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ") < %" PRIu64 ") {\n", lt->gl_LocalInvocationID_y, lt->firstStageStartSize, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->dim, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->localSize[1] * lt->firstStageStartSize, lt->fft_dim_full);
				res = AppendLineFromInput(lt, lt->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				sprintf(lt->disableThreadsEnd, "}");
			}
		}
		else {
			lt->tempLen = sprintf(lt->tempStr, "		{ \n");
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
		}

		if ((lt->localSize[1] > 1) || (0 && (lt->inverse)) || (lt->localSize[0] * lt->stageRadix[0] * (lt->threadRadixRegister[lt->stageRadix[0]] / lt->stageRadix[0]) > lt->dim))
			lt->readToRegisters = 0;
		else
			lt->readToRegisters = 1;
		if (lt->dim == lt->fft_dim_full) {
			for (uint64_t k = 0; k < lt->regAd; k++) {
				for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {

					if (lt->localSize[1] == 1)
						lt->tempLen = sprintf(lt->tempStr, "		combinedID = %s + %" PRIu64 ";\n", lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0]);
					else
						lt->tempLen = sprintf(lt->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", lt->gl_LocalInvocationID_x, lt->localSize[0], lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[0] * lt->localSize[1]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (lt->inputStride[0] > 1)
						lt->tempLen = sprintf(lt->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") * %" PRIu64 " + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", lt->dim, lt->inputStride[0], lt->dim, lt->inputStride[1]);
					else
						lt->tempLen = sprintf(lt->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", lt->dim, lt->dim, lt->inputStride[1]);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					if (lt->axisSwapped) {
						if (lt->size[lt->axis_id + 1] % lt->localSize[0] != 0) {
							lt->tempLen = sprintf(lt->tempStr, "		if(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 "){", lt->dim, lt->gl_WorkGroupID_y, shiftY2, lt->localSize[0], lt->size[lt->axis_id + 1]);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}
					else {
						if (lt->size[lt->axis_id + 1] % lt->localSize[1] != 0) {
							lt->tempLen = sprintf(lt->tempStr, "		if(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 "){", lt->dim, lt->gl_WorkGroupID_y, shiftY2, lt->localSize[1], lt->size[lt->axis_id + 1]);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}
					lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					res = indexInputFFT(lt, uintType, readType, lt->inoutID, 0, requestCoordinate, requestBatch);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, ";\n");
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					if (lt->readToRegisters) {
						if (lt->inputBufferBlockNum == 1)
							lt->tempLen = sprintf(lt->tempStr, "		%s = %s%s[%s]%s;\n", lt->regIDs[i + k * lt->threadRegister], convTypeLeft, inputsStruct, lt->inoutID, convTypeRight);
						else
							lt->tempLen = sprintf(lt->tempStr, "		%s = %sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", lt->regIDs[i + k * lt->threadRegister], convTypeLeft, lt->inoutID, lt->inputBufferBlockSize, inputsStruct, lt->inoutID, lt->inputBufferBlockSize, convTypeRight);
					}
					else {
						if (lt->axisSwapped) {
							if (lt->inputBufferBlockNum == 1)
								lt->tempLen = sprintf(lt->tempStr, "		sdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")] = %s%s[%s]%s;\n", lt->dim, lt->dim, convTypeLeft, inputsStruct, lt->inoutID, convTypeRight);
							else
								lt->tempLen = sprintf(lt->tempStr, "		sdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")] = %sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", lt->dim, lt->dim, convTypeLeft, lt->inoutID, lt->inputBufferBlockSize, inputsStruct, lt->inoutID, lt->inputBufferBlockSize, convTypeRight);
						}
						else {
							if (lt->inputBufferBlockNum == 1)
								lt->tempLen = sprintf(lt->tempStr, "		sdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride] = %s%s[%s]%s;\n", lt->dim, lt->dim, convTypeLeft, inputsStruct, lt->inoutID, convTypeRight);
							else
								lt->tempLen = sprintf(lt->tempStr, "		sdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride] = %sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", lt->dim, lt->dim, convTypeLeft, lt->inoutID, lt->inputBufferBlockSize, inputsStruct, lt->inoutID, lt->inputBufferBlockSize, convTypeRight);
						}
					}
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					if (lt->axisSwapped) {
						if (lt->size[lt->axis_id + 1] % lt->localSize[0] != 0) {
							lt->tempLen = sprintf(lt->tempStr, "		}");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}
					else {
						if (lt->size[lt->axis_id + 1] % lt->localSize[1] != 0) {
							lt->tempLen = sprintf(lt->tempStr, "		}");
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}

				}
			}

		}
		else {
			for (uint64_t k = 0; k < lt->regAd; k++) {
				for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {

					if (lt->axisSwapped) {
						if (lt->localSize[1] == 1)
							lt->tempLen = sprintf(lt->tempStr, "		combinedID = %s + %" PRIu64 ";\n", lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0]);
						else
							lt->tempLen = sprintf(lt->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 "*numActiveThreads;\n", lt->gl_LocalInvocationID_x, lt->localSize[0], lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin));
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
						lt->tempLen = sprintf(lt->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ");\n", lt->dim, lt->dim, lt->firstStageStartSize, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->dim, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->localSize[0] * lt->firstStageStartSize);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						lt->tempLen = sprintf(lt->tempStr, "		inoutID = %s+%" PRIu64 "+%s * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ");\n", lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0], lt->gl_LocalInvocationID_y, lt->firstStageStartSize, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->dim, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->localSize[1] * lt->firstStageStartSize);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
					lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					res = indexInputFFT(lt, uintType, readType, lt->inoutID, 0, requestCoordinate, requestBatch);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, ";\n");
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					if (lt->readToRegisters) {
						if (lt->inputBufferBlockNum == 1)
							lt->tempLen = sprintf(lt->tempStr, "			%s = %s%s[%s]%s;\n", lt->regIDs[i + k * lt->threadRegister], convTypeLeft, inputsStruct, lt->inoutID, convTypeRight);
						else
							lt->tempLen = sprintf(lt->tempStr, "			%s = %sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", lt->regIDs[i + k * lt->threadRegister], convTypeLeft, lt->inoutID, lt->inputBufferBlockSize, inputsStruct, lt->inoutID, lt->inputBufferBlockSize, convTypeRight);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						if (lt->axisSwapped) {

							if (lt->inputBufferBlockNum == 1)
								lt->tempLen = sprintf(lt->tempStr, "		sdata[(combinedID / %" PRIu64 ") + sharedStride*(combinedID %% %" PRIu64 ")] = %s%s[inoutID]%s;\n", lt->dim, lt->dim, convTypeLeft, inputsStruct, convTypeRight);
							else
								lt->tempLen = sprintf(lt->tempStr, "		sdata[(combinedID / %" PRIu64 ") + sharedStride*(combinedID %% %" PRIu64 ")] = %sinputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "]%s;\n", lt->dim, lt->dim, convTypeLeft, lt->inputBufferBlockSize, inputsStruct, lt->inputBufferBlockSize, convTypeRight);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (lt->inputBufferBlockNum == 1)
								lt->tempLen = sprintf(lt->tempStr, "		sdata[sharedStride*%s + (%s + %" PRIu64 ")] = %s%s[inoutID]%s;\n", lt->gl_LocalInvocationID_y, lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0], convTypeLeft, inputsStruct, convTypeRight);
							else
								lt->tempLen = sprintf(lt->tempStr, "		sdata[sharedStride*%s + (%s + %" PRIu64 ")] = %sinputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "]%s;\n", lt->gl_LocalInvocationID_y, lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0], convTypeLeft, lt->inputBufferBlockSize, inputsStruct, lt->inputBufferBlockSize, convTypeRight);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
					}
				}
			}
		}
		lt->tempLen = sprintf(lt->tempStr, "	}\n");
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 1://grouped_c2c
	{
		if (lt->localSize[1] * lt->stageRadix[0] * (lt->threadRadixRegister[lt->stageRadix[0]] / lt->stageRadix[0]) > lt->dim)
			lt->readToRegisters = 0;
		else
			lt->readToRegisters = 1;
		char shiftX[500] = "";
		if (lt->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", lt->gl_WorkGroupSize_x);

		sprintf(lt->disableThreadsStart, "		if (((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x, lt->stageStartSize, lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x * lt->stageStartSize, lt->dim * lt->stageStartSize, lt->size[lt->axis_id]);
		res = AppendLineFromInput(lt, lt->disableThreadsStart);
		if (res != FFT_SUCCESS) return res;
		sprintf(lt->disableThreadsEnd, "}");
		for (uint64_t k = 0; k < lt->regAd; k++) {
			for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
				lt->tempLen = sprintf(lt->tempStr, "		inoutID = (%" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 "));\n", lt->stageStartSize, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x, lt->stageStartSize, lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x * lt->stageStartSize, lt->dim * lt->stageStartSize);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
				lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
				sprintf(index_x, "(%s%s) %% (%" PRIu64 ")", lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x);
				res = indexInputFFT(lt, uintType, readType, index_x, lt->inoutID, requestCoordinate, requestBatch);
				if (res != FFT_SUCCESS) return res;
				lt->tempLen = sprintf(lt->tempStr, ";\n");
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;

				if (lt->readToRegisters) {
					if (lt->inputBufferBlockNum == 1)
						lt->tempLen = sprintf(lt->tempStr, "			%s=%s%s[%s]%s;\n", lt->regIDs[i + k * lt->threadRegister], convTypeLeft, inputsStruct, lt->inoutID, convTypeRight);
					else
						lt->tempLen = sprintf(lt->tempStr, "			%s=%sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", lt->regIDs[i + k * lt->threadRegister], convTypeLeft, lt->inoutID, lt->inputBufferBlockSize, inputsStruct, lt->inoutID, lt->inputBufferBlockSize, convTypeRight);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					if (lt->inputBufferBlockNum == 1)
						lt->tempLen = sprintf(lt->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s]=%s%s[%s]%s;\n", lt->sharedStride, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_LocalInvocationID_x, convTypeLeft, inputsStruct, lt->inoutID, convTypeRight);
					else
						lt->tempLen = sprintf(lt->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s]=%sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", lt->sharedStride, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_LocalInvocationID_x, convTypeLeft, lt->inoutID, lt->inputBufferBlockSize, inputsStruct, lt->inoutID, lt->inputBufferBlockSize, convTypeRight);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}

			}
		}
		lt->tempLen = sprintf(lt->tempStr, "	}\n");
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 2://single_c2c_strided
	{
		if (lt->localSize[1] * lt->stageRadix[0] * (lt->threadRadixRegister[lt->stageRadix[0]] / lt->stageRadix[0]) > lt->dim)
			lt->readToRegisters = 0;
		else
			lt->readToRegisters = 1;
		char shiftX[500] = "";
		if (lt->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", lt->gl_WorkGroupSize_x);

		//lt->tempLen = sprintf(lt->tempStr, "		if(gl_GlobalInvolcationID.x%s >= %" PRIu64 ") return; \n", shiftX, lt->size[0] / axis->layout.dim);
		sprintf(lt->disableThreadsStart, "		if (((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_x, shiftX, lt->stageStartSize, lt->stageStartSize * lt->dim, lt->fft_dim_full);
		res = AppendLineFromInput(lt, lt->disableThreadsStart);
		if (res != FFT_SUCCESS) return res;
		sprintf(lt->disableThreadsEnd, "}");
		for (uint64_t k = 0; k < lt->regAd; k++) {
			for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
				lt->tempLen = sprintf(lt->tempStr, "		inoutID = (%s%s) %% (%" PRIu64 ") + %" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") * (%" PRIu64 ");\n", lt->gl_GlobalInvocationID_x, shiftX, lt->stageStartSize, lt->stageStartSize, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_GlobalInvocationID_x, shiftX, lt->stageStartSize, lt->stageStartSize * lt->dim);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
				lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
				res = indexInputFFT(lt, uintType, readType, lt->inoutID, 0, requestCoordinate, requestBatch);
				if (res != FFT_SUCCESS) return res;
				lt->tempLen = sprintf(lt->tempStr, ";\n");
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;

				if (lt->readToRegisters) {
					if (lt->inputBufferBlockNum == 1)
						lt->tempLen = sprintf(lt->tempStr, "			%s=%s%s[%s]%s;\n", lt->regIDs[i + k * lt->threadRegister], convTypeLeft, inputsStruct, lt->inoutID, convTypeRight);
					else
						lt->tempLen = sprintf(lt->tempStr, "			%s=%sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", lt->regIDs[i + k * lt->threadRegister], convTypeLeft, lt->inoutID, lt->inputBufferBlockSize, inputsStruct, lt->inoutID, lt->inputBufferBlockSize, convTypeRight);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					if (lt->inputBufferBlockNum == 1)
						lt->tempLen = sprintf(lt->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s]=%s%s[%s]%s;\n", lt->sharedStride, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_LocalInvocationID_x, convTypeLeft, inputsStruct, lt->inoutID, convTypeRight);
					else
						lt->tempLen = sprintf(lt->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s]=%sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", lt->sharedStride, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_LocalInvocationID_x, convTypeLeft, lt->inoutID, lt->inputBufferBlockSize, inputsStruct, lt->inoutID, lt->inputBufferBlockSize, convTypeRight);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
			}
		}
		lt->tempLen = sprintf(lt->tempStr, "	}\n");
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	
	
	}
	return res;
}


FFTResult appendWriteDataFFT(FFTLayout* lt, const char* floatType, const char* floatTypeMemory, const char* uintType, uint64_t writeType) {
	FFTResult res = FFT_SUCCESS;
	double double_PI = 3.1415926535897932384626433832795;
	char vecType[30];
	char outputsStruct[20] = "";
	char LFending[4] = "";

	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	sprintf(outputsStruct, "outputs");
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");
	char cosDef[20] = "__cosf";
	char sinDef[20] = "__sinf";

	char convTypeLeft[20] = "";
	char convTypeRight[20] = "";
	if ((!strcmp(floatTypeMemory, "half")) && (strcmp(floatType, "half"))) {
		if ((writeType == 6) || (writeType == 120) || (writeType == 121) || (writeType == 130) || (writeType == 131) || (writeType == 140) || (writeType == 141) || (writeType == 142) || (writeType == 143)) {
			sprintf(convTypeLeft, "float16_t(");
			sprintf(convTypeRight, ")");
		}
		else {
			sprintf(convTypeLeft, "f16vec2(");
			sprintf(convTypeRight, ")");
		}
	}

	if ((!strcmp(floatTypeMemory, "double")) && (strcmp(floatType, "double"))) {
		if ((writeType == 6) || (writeType == 120) || (writeType == 121) || (writeType == 130) || (writeType == 131) || (writeType == 140) || (writeType == 141) || (writeType == 142) || (writeType == 143)) {

			sprintf(convTypeLeft, "(double)");
			//sprintf(convTypeRight, "");

		}
		else {

			sprintf(convTypeLeft, "conv_double2(");
			sprintf(convTypeRight, ")");

		}
	}

	char index_x[2000] = "";
	char index_y[2000] = "";
	char requestCoordinate[100] = "";
	if (lt->convolutionStep) {
		if (lt->matrixConvolution > 1) {
			sprintf(requestCoordinate, "coordinate");
		}
	}
	char requestBatch[100] = "";
	if (lt->convolutionStep) {
		if (lt->numKernels > 1) {
			sprintf(requestBatch, "batchID");//if one buffer - multiple kernel convolution
		}
	}
	switch (writeType) {
	case 0: //single_c2c
	{
		if ((lt->localSize[1] > 1) || (lt->localSize[0] * lt->stageRadix[lt->numStages - 1] * (lt->threadRadixRegister[lt->stageRadix[lt->numStages - 1]] / lt->stageRadix[lt->numStages - 1]) > lt->dim)) {
			lt->writeFromRegisters = 0;
			res = appendBarrier(lt, 1);
			if (res != FFT_SUCCESS) return res;
		}
		else
			lt->writeFromRegisters = 1;
		char shiftX[500] = "";
		if (lt->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX ");
		char shiftY[500] = "";
		if (lt->axisSwapped) {
			if (lt->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*%s ", lt->gl_WorkGroupSize_x);
		}
		else {
			if (lt->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*%s ", lt->gl_WorkGroupSize_y);
		}

		char shiftY2[100] = "";
		if (lt->performWorkGroupShift[1])
			sprintf(shiftY, " + consts.workGroupShiftY ");
		if (lt->dim < lt->fft_dim_full) {
			if (lt->axisSwapped) {
				if (!lt->reorderFourStep) {
					lt->tempLen = sprintf(lt->tempStr, "		if((%s+%" PRIu64 "*%s)< numActiveThreads) {\n", lt->gl_LocalInvocationID_x, lt->localSize[0], lt->gl_LocalInvocationID_y);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					lt->tempLen = sprintf(lt->tempStr, "		if (((%s + %" PRIu64 " * %s) %% %" PRIu64 " + ((%s%s) / %" PRIu64 ")*%" PRIu64 " < %" PRIu64 ")){\n", lt->gl_LocalInvocationID_x, lt->localSize[0], lt->gl_LocalInvocationID_y, lt->localSize[0], lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->localSize[0], lt->fft_dim_full / lt->firstStageStartSize);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
			}
			else {
				lt->tempLen = sprintf(lt->tempStr, "		if (((%s + %" PRIu64 " * %s) %% %" PRIu64 " + ((%s%s) / %" PRIu64 ")*%" PRIu64 " < %" PRIu64 ")){\n", lt->gl_LocalInvocationID_x, lt->localSize[0], lt->gl_LocalInvocationID_y, lt->localSize[1], lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->localSize[1], lt->fft_dim_full / lt->firstStageStartSize);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			lt->tempLen = sprintf(lt->tempStr, "		{ \n");
			res = AppendLine(lt);
			if (res != FFT_SUCCESS) return res;
		}


		if (lt->reorderFourStep) {

			if (lt->dim == lt->fft_dim_full) {
				for (uint64_t k = 0; k < lt->regAd; k++) {
					for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
						if (lt->localSize[1] == 1)
							lt->tempLen = sprintf(lt->tempStr, "		combinedID = %s + %" PRIu64 ";\n", lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0]);
						else
							lt->tempLen = sprintf(lt->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", lt->gl_LocalInvocationID_x, lt->localSize[0], lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[0] * lt->localSize[1]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;

						if (lt->outputStride[0] > 1)
							lt->tempLen = sprintf(lt->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") * %" PRIu64 " + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", lt->dim, lt->outputStride[0], lt->dim, lt->outputStride[1]);
						else
							lt->tempLen = sprintf(lt->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", lt->dim, lt->dim, lt->outputStride[1]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
						if (lt->axisSwapped) {
							if (lt->size[lt->axis_id + 1] % lt->localSize[0] != 0) {
								lt->tempLen = sprintf(lt->tempStr, "		if(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 "){", lt->dim, lt->gl_WorkGroupID_y, shiftY2, lt->localSize[0], lt->size[lt->axis_id + 1]);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}
						else {
							if (lt->size[lt->axis_id + 1] % lt->localSize[1] != 0) {
								lt->tempLen = sprintf(lt->tempStr, "		if(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 "){", lt->dim, lt->gl_WorkGroupID_y, shiftY2, lt->localSize[1], lt->size[lt->axis_id + 1]);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}

						lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;

						res = indexOutputFFT(lt, uintType, writeType, lt->inoutID, 0, requestCoordinate, requestBatch);
						if (res != FFT_SUCCESS) return res;
						lt->tempLen = sprintf(lt->tempStr, ";\n");
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;

						if (lt->writeFromRegisters) {
							if (lt->outputBufferBlockNum == 1)
								lt->tempLen = sprintf(lt->tempStr, "		%s[%s] = %s%s%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
							else
								lt->tempLen = sprintf(lt->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %s%s%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (lt->axisSwapped) {
								if (lt->outputBufferBlockNum == 1)
									lt->tempLen = sprintf(lt->tempStr, "		%s[%s] = %ssdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")]%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->dim, lt->dim, convTypeRight);
								else
									lt->tempLen = sprintf(lt->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")]%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->dim, lt->dim, convTypeRight);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
							else {
								if (lt->outputBufferBlockNum == 1)
									lt->tempLen = sprintf(lt->tempStr, "		%s[%s] = %ssdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride]%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->dim, lt->dim, convTypeRight);
								else
									lt->tempLen = sprintf(lt->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride]%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->dim, lt->dim, convTypeRight);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}

						if (lt->axisSwapped) {
							if (lt->size[lt->axis_id + 1] % lt->localSize[0] != 0) {
								lt->tempLen = sprintf(lt->tempStr, "		}");
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}
						else {
							if (lt->size[lt->axis_id + 1] % lt->localSize[1] != 0) {
								lt->tempLen = sprintf(lt->tempStr, "		}");
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}
					}
				}
			}
			else {
				for (uint64_t k = 0; k < lt->regAd; k++) {
					for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
						if (lt->localSize[1] == 1)
							lt->tempLen = sprintf(lt->tempStr, "		combinedID = %s + %" PRIu64 ";\n", lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0]);
						else
							lt->tempLen = sprintf(lt->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", lt->gl_LocalInvocationID_x, lt->localSize[0], lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[0] * lt->localSize[1]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
						if (lt->axisSwapped) {
							lt->tempLen = sprintf(lt->tempStr, "		inoutID = combinedID %% %" PRIu64 " + ((%s%s) / %" PRIu64 ")*%" PRIu64 " + ((combinedID/%" PRIu64 ") * %" PRIu64 ")+ ((%s%s) %% %" PRIu64 ") * %" PRIu64 ";\n", lt->localSize[0], lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->localSize[0], lt->localSize[0], lt->fft_dim_full / lt->dim, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->fft_dim_full / lt->firstStageStartSize);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (lt->localSize[1] == 1)
								lt->tempLen = sprintf(lt->tempStr, "		inoutID = (%s%s)/%" PRIu64 "+ (combinedID * %" PRIu64 ")+ ((%s%s) %% %" PRIu64 ") * %" PRIu64 ";\n", lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->fft_dim_full / lt->dim, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->fft_dim_full / lt->firstStageStartSize);
							else
								lt->tempLen = sprintf(lt->tempStr, "		inoutID = combinedID %% %" PRIu64 " + ((%s%s) / %" PRIu64 ")*%" PRIu64 " + ((combinedID/%" PRIu64 ") * %" PRIu64 ")+ ((%s%s) %% %" PRIu64 ") * %" PRIu64 ";\n", lt->localSize[1], lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->localSize[1], lt->localSize[1], lt->fft_dim_full / lt->dim, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->fft_dim_full / lt->firstStageStartSize);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}

						lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
						res = indexOutputFFT(lt, uintType, writeType, lt->inoutID, 0, requestCoordinate, requestBatch);
						if (res != FFT_SUCCESS) return res;
						lt->tempLen = sprintf(lt->tempStr, ";\n");
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;

						if (lt->writeFromRegisters) {
							if (lt->outputBufferBlockNum == 1)
								lt->tempLen = sprintf(lt->tempStr, "			%s[%s] = %s%s%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
							else
								lt->tempLen = sprintf(lt->tempStr, "			outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %s%s%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (lt->axisSwapped) {
								if (lt->outputBufferBlockNum == 1)
									lt->tempLen = sprintf(lt->tempStr, "			%s[%s] = %ssdata[(combinedID %% %s)+(combinedID/%s)*sharedStride]%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->gl_WorkGroupSize_x, lt->gl_WorkGroupSize_x, convTypeRight);
								else
									lt->tempLen = sprintf(lt->tempStr, "			outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %s)+(combinedID/%s)*sharedStride]%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->gl_WorkGroupSize_x, lt->gl_WorkGroupSize_x, convTypeRight);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
							else {
								if (lt->outputBufferBlockNum == 1)
									lt->tempLen = sprintf(lt->tempStr, "			%s[%s] = %ssdata[(combinedID %% %s)*sharedStride+combinedID/%s]%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->gl_WorkGroupSize_y, lt->gl_WorkGroupSize_y, convTypeRight);
								else
									lt->tempLen = sprintf(lt->tempStr, "			outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %s)*sharedStride+combinedID/%s]%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->gl_WorkGroupSize_y, lt->gl_WorkGroupSize_y, convTypeRight);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}
					}
				}
			}
		}

		else {
			if (lt->dim == lt->fft_dim_full) {
				for (uint64_t k = 0; k < lt->regAd; k++) {
					for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
						if (lt->localSize[1] == 1)
							lt->tempLen = sprintf(lt->tempStr, "		combinedID = %s + %" PRIu64 ";\n", lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0]);
						else
							lt->tempLen = sprintf(lt->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", lt->gl_LocalInvocationID_x, lt->localSize[0], lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[0] * lt->localSize[1]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;

						if (lt->outputStride[0] > 1)
							lt->tempLen = sprintf(lt->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") * %" PRIu64 " + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", lt->dim, lt->outputStride[0], lt->dim, lt->outputStride[1]);
						else
							lt->tempLen = sprintf(lt->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", lt->dim, lt->dim, lt->outputStride[1]);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
						if (lt->axisSwapped) {
							if (lt->size[lt->axis_id + 1] % lt->localSize[0] != 0) {
								lt->tempLen = sprintf(lt->tempStr, "		if(combinedID / %" PRIu64 " + %s*%" PRIu64 "< %" PRIu64 "){", lt->dim, lt->gl_WorkGroupID_y, lt->localSize[0], lt->size[lt->axis_id + 1]);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}
						else {
							if (lt->size[lt->axis_id + 1] % lt->localSize[1] != 0) {
								lt->tempLen = sprintf(lt->tempStr, "		if(combinedID / %" PRIu64 " + %s*%" PRIu64 "< %" PRIu64 "){", lt->dim, lt->gl_WorkGroupID_y, lt->localSize[1], lt->size[lt->axis_id + 1]);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}

						lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;

						res = indexOutputFFT(lt, uintType, writeType, lt->inoutID, 0, requestCoordinate, requestBatch);

						if (res != FFT_SUCCESS) return res;
						lt->tempLen = sprintf(lt->tempStr, ";\n");
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;

						if (lt->writeFromRegisters) {
							if (lt->outputBufferBlockNum == 1)
								lt->tempLen = sprintf(lt->tempStr, "		%s[%s] = %s%s%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
							else
								lt->tempLen = sprintf(lt->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %s%s%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (lt->axisSwapped) {
								if (lt->outputBufferBlockNum == 1)
									lt->tempLen = sprintf(lt->tempStr, "		%s[%s] = %ssdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")]%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->dim, lt->dim, convTypeRight);
								else
									lt->tempLen = sprintf(lt->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")]%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->dim, lt->dim, convTypeRight);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
							else {
								if (lt->outputBufferBlockNum == 1)
									lt->tempLen = sprintf(lt->tempStr, "		%s[%s] = %ssdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride]%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->dim, lt->dim, convTypeRight);
								else
									lt->tempLen = sprintf(lt->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride]%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->dim, lt->dim, convTypeRight);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}
						if (lt->axisSwapped) {
							if (lt->size[lt->axis_id + 1] % lt->localSize[0] != 0) {
								lt->tempLen = sprintf(lt->tempStr, "		}");
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}
						else {
							if (lt->size[lt->axis_id + 1] % lt->localSize[1] != 0) {
								lt->tempLen = sprintf(lt->tempStr, "		}");
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}
					}
				}
			}
			else {
				for (uint64_t k = 0; k < lt->regAd; k++) {
					for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
						if (lt->localSize[1] == 1)
							lt->tempLen = sprintf(lt->tempStr, "		combinedID = %s + %" PRIu64 ";\n", lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0]);
						else
							lt->tempLen = sprintf(lt->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 " * numActiveThreads;\n", lt->gl_LocalInvocationID_x, lt->localSize[0], lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin));
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
						if (lt->axisSwapped) {
							lt->tempLen = sprintf(lt->tempStr, "		inoutID = (combinedID %% %" PRIu64 ")+(combinedID / %" PRIu64 ") * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ");", lt->dim, lt->dim, lt->firstStageStartSize, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->dim, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->localSize[0] * lt->firstStageStartSize);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							lt->tempLen = sprintf(lt->tempStr, "		inoutID = %s+%" PRIu64 "+%s * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ");", lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0], lt->gl_LocalInvocationID_y, lt->firstStageStartSize, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->dim, lt->gl_WorkGroupID_x, shiftX, lt->firstStageStartSize / lt->dim, lt->localSize[1] * lt->firstStageStartSize);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}

						lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
						res = indexOutputFFT(lt, uintType, writeType, lt->inoutID, 0, requestCoordinate, requestBatch);
						if (res != FFT_SUCCESS) return res;
						lt->tempLen = sprintf(lt->tempStr, ";\n");
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;

						if (lt->writeFromRegisters) {
							if (lt->outputBufferBlockNum == 1)
								lt->tempLen = sprintf(lt->tempStr, "		%s[inoutID]=%s%s%s;\n", outputsStruct, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
							else
								lt->tempLen = sprintf(lt->tempStr, "		outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] = %s%s%s;\n", lt->outputBufferBlockSize, outputsStruct, lt->outputBufferBlockSize, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
							res = AppendLine(lt);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (lt->axisSwapped) {
								if (lt->outputBufferBlockNum == 1)
									lt->tempLen = sprintf(lt->tempStr, "		%s[inoutID]=%ssdata[%s + sharedStride*(%s + %" PRIu64 ")]%s;\n", outputsStruct, convTypeLeft, lt->gl_LocalInvocationID_x, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], convTypeRight);
								else
									lt->tempLen = sprintf(lt->tempStr, "		outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] = %ssdata[%s + sharedStride*(%s + %" PRIu64 ")]%s;\n", lt->outputBufferBlockSize, outputsStruct, lt->outputBufferBlockSize, convTypeLeft, lt->gl_LocalInvocationID_x, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], convTypeRight);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
							else {
								if (lt->outputBufferBlockNum == 1)
									lt->tempLen = sprintf(lt->tempStr, "		%s[inoutID]=%ssdata[sharedStride*%s + (%s + %" PRIu64 ")]%s;\n", outputsStruct, convTypeLeft, lt->gl_LocalInvocationID_y, lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0], convTypeRight);
								else
									lt->tempLen = sprintf(lt->tempStr, "		outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] = %ssdata[sharedStride*%s + (%s + %" PRIu64 ")]%s;\n", lt->outputBufferBlockSize, outputsStruct, lt->outputBufferBlockSize, convTypeLeft, lt->gl_LocalInvocationID_y, lt->gl_LocalInvocationID_x, (i + k * lt->threadRegisterMin) * lt->localSize[0], convTypeRight);
								res = AppendLine(lt);
								if (res != FFT_SUCCESS) return res;
							}
						}
					}
				}
			}
		}

		lt->tempLen = sprintf(lt->tempStr, "	}\n");
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 1: //grouped_c2c
	{
		if (lt->localSize[1] * lt->stageRadix[lt->numStages - 1] * (lt->threadRadixRegister[lt->stageRadix[lt->numStages - 1]] / lt->stageRadix[lt->numStages - 1]) > lt->dim) {
			lt->writeFromRegisters = 0;
			res = appendBarrier(lt, 1);
			if (res != FFT_SUCCESS) return res;
		}
		else
			lt->writeFromRegisters = 1;
		char shiftX[500] = "";
		if (lt->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", lt->gl_WorkGroupSize_x);
		lt->tempLen = sprintf(lt->tempStr, "		if (((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x, lt->stageStartSize, lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x * lt->stageStartSize, lt->dim * lt->stageStartSize, lt->size[lt->axis_id]);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		if ((lt->reorderFourStep) && (lt->stageStartSize == 1)) {
			for (uint64_t k = 0; k < lt->regAd; k++) {
				for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
					lt->tempLen = sprintf(lt->tempStr, "		inoutID = (%s + %" PRIu64 ") * (%" PRIu64 ") + (((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")) * (%" PRIu64 ") + ((%s%s) / %" PRIu64 ");\n", lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->fft_dim_full / lt->dim, lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x, lt->firstStageStartSize / lt->dim, lt->fft_dim_full / lt->firstStageStartSize, lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x * (lt->firstStageStartSize / lt->dim));
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					sprintf(index_x, "(%s%s) %% (%" PRIu64 ")", lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x);
					res = indexOutputFFT(lt, uintType, writeType, index_x, lt->inoutID, requestCoordinate, requestBatch);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, ";\n");
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					if (lt->writeFromRegisters) {
						if (lt->outputBufferBlockNum == 1)
							lt->tempLen = sprintf(lt->tempStr, "			%s[%s] = %s%s%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
						else
							lt->tempLen = sprintf(lt->tempStr, "			outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %s%s%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						if (lt->outputBufferBlockNum == 1)
							lt->tempLen = sprintf(lt->tempStr, "			%s[%s] = %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", outputsStruct, lt->inoutID, convTypeLeft, lt->sharedStride, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_LocalInvocationID_x, convTypeRight);
						else
							lt->tempLen = sprintf(lt->tempStr, "			outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", lt->inoutID, lt->outputBufferBlockSize, outputsStruct, lt->inoutID, lt->outputBufferBlockSize, convTypeLeft, lt->sharedStride, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_LocalInvocationID_x, convTypeRight);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;

					}
				}
			}
		}
		else {
			for (uint64_t k = 0; k < lt->regAd; k++) {
				for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {

					lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
					sprintf(index_x, "(%s%s) %% (%" PRIu64 ")", lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x);
					sprintf(index_y, "%" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ")", lt->stageStartSize, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x, lt->stageStartSize, lt->gl_GlobalInvocationID_x, shiftX, lt->fft_dim_x * lt->stageStartSize, lt->stageStartSize * lt->dim);
					res = indexOutputFFT(lt, uintType, writeType, index_x, index_y, requestCoordinate, requestBatch);
					if (res != FFT_SUCCESS) return res;
					lt->tempLen = sprintf(lt->tempStr, ";\n");
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;

					if (lt->writeFromRegisters) {
						if (lt->outputBufferBlockNum == 1)
							lt->tempLen = sprintf(lt->tempStr, "			%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
						else
							lt->tempLen = sprintf(lt->tempStr, "			outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] =  %s%s%s;\n", lt->outputBufferBlockSize, outputsStruct, lt->outputBufferBlockSize, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						if (lt->outputBufferBlockNum == 1)
							lt->tempLen = sprintf(lt->tempStr, "			%s[inoutID] = %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", outputsStruct, convTypeLeft, lt->sharedStride, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_LocalInvocationID_x, convTypeRight);
						else
							lt->tempLen = sprintf(lt->tempStr, "			outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] =  %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", lt->outputBufferBlockSize, outputsStruct, lt->outputBufferBlockSize, convTypeLeft, lt->sharedStride, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_LocalInvocationID_x, convTypeRight);
						res = AppendLine(lt);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
		}
		lt->tempLen = sprintf(lt->tempStr, "	}\n");
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		break;

	}
	case 2: //single_c2c_strided
	{
		if (lt->localSize[1] * lt->stageRadix[lt->numStages - 1] * (lt->threadRadixRegister[lt->stageRadix[lt->numStages - 1]] / lt->stageRadix[lt->numStages - 1]) > lt->dim) {
			lt->writeFromRegisters = 0;
			res = appendBarrier(lt, 1);
			if (res != FFT_SUCCESS) return res;
		}
		else
			lt->writeFromRegisters = 1;
		char shiftX[500] = "";
		if (lt->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", lt->gl_WorkGroupSize_x);
		lt->tempLen = sprintf(lt->tempStr, "		if (((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") {\n", lt->gl_GlobalInvocationID_x, shiftX, lt->stageStartSize, lt->stageStartSize * lt->dim, lt->fft_dim_full);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		for (uint64_t k = 0; k < lt->regAd; k++) {
			for (uint64_t i = 0; i < lt->threadRegisterMin; i++) {
				lt->tempLen = sprintf(lt->tempStr, "		inoutID = (%s%s) %% (%" PRIu64 ") + %" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") * (%" PRIu64 ");\n", lt->gl_GlobalInvocationID_x, shiftX, lt->stageStartSize, lt->stageStartSize, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_GlobalInvocationID_x, shiftX, lt->stageStartSize, lt->stageStartSize * lt->dim);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
				lt->tempLen = sprintf(lt->tempStr, "			%s = ", lt->inoutID);
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;
				res = indexOutputFFT(lt, uintType, writeType, lt->inoutID, 0, requestCoordinate, requestBatch);
				if (res != FFT_SUCCESS) return res;
				lt->tempLen = sprintf(lt->tempStr, ";\n");
				res = AppendLine(lt);
				if (res != FFT_SUCCESS) return res;

				if (lt->writeFromRegisters) {
					if (lt->outputBufferBlockNum == 1)
						lt->tempLen = sprintf(lt->tempStr, "			%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
					else
						lt->tempLen = sprintf(lt->tempStr, "			outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] = %s%s%s;\n", lt->outputBufferBlockSize, outputsStruct, lt->outputBufferBlockSize, convTypeLeft, lt->regIDs[i + k * lt->threadRegister], convTypeRight);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					if (lt->outputBufferBlockNum == 1)
						lt->tempLen = sprintf(lt->tempStr, "			%s[inoutID] = %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", outputsStruct, convTypeLeft, lt->sharedStride, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_LocalInvocationID_x, convTypeRight);
					else
						lt->tempLen = sprintf(lt->tempStr, "			outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] = %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", lt->outputBufferBlockSize, outputsStruct, lt->outputBufferBlockSize, convTypeLeft, lt->sharedStride, lt->gl_LocalInvocationID_y, (i + k * lt->threadRegisterMin) * lt->localSize[1], lt->gl_LocalInvocationID_x, convTypeRight);
					res = AppendLine(lt);
					if (res != FFT_SUCCESS) return res;
				}
			}
		}
		lt->tempLen = sprintf(lt->tempStr, "	}\n");
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) return res;
		break;

	}
	}
	return res;
}



FFTResult FFTScheduler(FFTApplication* app, FFTPlan* FFTPlan, uint64_t axis_id, uint64_t supportAxis) {

	FFTAxis* axes = FFTPlan->axes[axis_id];
	uint64_t complexSize;
	if (app->configuration.doublePrecision) complexSize = (2 * sizeof(double));
	uint64_t maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / complexSize;
	uint64_t maxSingleSizeNonStrided = maxSequenceLengthSharedMemory;
	uint64_t nonStridedAxisId = 0;
	for (uint64_t i = 0; i < 3; i++) {
		FFTPlan->actualFFTSizePerAxis[axis_id][i] = app->configuration.size[i];
	}

	uint64_t multipliers[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };//split the sequence
	uint64_t isPowOf2 = (pow(2, (uint64_t)log2(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id])) == FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]) ? 1 : 0;
	uint64_t tempSequence = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
	for (uint64_t i = 2; i < 14; i++) {
		if (tempSequence % i == 0) {
			tempSequence /= i;
			multipliers[i]++;
			i--;
		}
	}
	if (tempSequence != 1) return FFT_ERROR_UNSUPPORTED_RADIX;
	uint64_t regAd = 1;
	for (uint64_t i = 1; i <= app->configuration.regAd; i++) {
		if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0)
			regAd = i;
	}
	if (axis_id == nonStridedAxisId) maxSingleSizeNonStrided *= regAd;
	uint64_t maxSequenceLengthSharedMemoryStrided = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySize / complexSize;
	uint64_t maxSingleSizeStrided = (!0) ? maxSequenceLengthSharedMemoryStrided * regAd : maxSequenceLengthSharedMemoryStrided;
	uint64_t numPasses = 1;
	uint64_t numPassesHalfBandwidth = 1;
	uint64_t temp;
	temp = (axis_id == nonStridedAxisId) ? (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeNonStrided) : (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeStrided);
	if (temp > 1) {//more passes than one
		for (uint64_t i = 1; i <= 1; i++) {
			if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0) {
				regAd = i;
			}
		}
		maxSingleSizeNonStrided = maxSequenceLengthSharedMemory * regAd;
		maxSingleSizeStrided = maxSequenceLengthSharedMemoryStrided * regAd;
		temp = (axis_id == nonStridedAxisId) ? FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeNonStrided : FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeStrided;
        numPasses = (uint64_t)ceil(log2(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]) / log2(maxSingleSizeStrided));
		//numPasses += (uint64_t)ceil(log2(temp) / log2(maxSingleSizeStrided));
	}
	regAd = ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep) || (numPasses == 1))) ? (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)(pow(maxSequenceLengthSharedMemoryStrided, numPasses - 1) * maxSequenceLengthSharedMemory)) : (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)pow(maxSequenceLengthSharedMemoryStrided, numPasses));
	uint64_t canBoost = 0;
	for (uint64_t i = regAd; i <= app->configuration.regAd; i++) {
		if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0) {
			regAd = i;
			i = app->configuration.regAd + 1;
			canBoost = 1;
		}
	}
	if (((canBoost == 0) || (((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] & (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1)) != 0))) && (regAd > 1)) {
		regAd = 1;
		numPasses++;
	}
	maxSingleSizeNonStrided = maxSequenceLengthSharedMemory * regAd;
	maxSingleSizeStrided = maxSequenceLengthSharedMemoryStrided * regAd;
	uint64_t maxSingleSizeStridedHalfBandwidth = maxSingleSizeStrided;

    
	uint64_t* locAxisSplit = FFTPlan->axisSplit[axis_id];
	if (numPasses == 1) {
		locAxisSplit[0] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
	}
	if (numPasses == 2) {
		if (isPowOf2) {

			uint64_t maxPow8Strided = (uint64_t)pow(8, ((uint64_t)log2(maxSingleSizeStrided)) / 3);
				//all FFTs are considered as non-unit stride
			if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxPow8Strided <= maxSingleSizeStrided) {
				locAxisSplit[0] = maxPow8Strided;
			}
			else {
				if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeStrided < maxSingleSizeStridedHalfBandwidth) {
					locAxisSplit[0] = maxSingleSizeStrided;
				}
				else {
					locAxisSplit[0] = maxSingleSizeStridedHalfBandwidth;
				}
			}
			locAxisSplit[1] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[0];
			if (locAxisSplit[1] < 64) {
				locAxisSplit[0] = (locAxisSplit[1] == 0) ? locAxisSplit[0] / (64) : locAxisSplit[0] / (64 / locAxisSplit[1]);
				locAxisSplit[1] = 64;
			}
			if (locAxisSplit[1] > locAxisSplit[0]) {
				uint64_t swap = locAxisSplit[0];
				locAxisSplit[0] = locAxisSplit[1];
				locAxisSplit[1] = swap;
			}
		}
		else {
			uint64_t successSplit = 0;
			uint64_t sqrtSequence = (uint64_t)ceil(sqrt(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]));
			for (uint64_t i = 0; i < sqrtSequence; i++) {
				if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (sqrtSequence - i) == 0) {
					if ((sqrtSequence - i <= maxSingleSizeStrided) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrtSequence - i) <= maxSingleSizeStridedHalfBandwidth)) {
						locAxisSplit[0] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrtSequence - i);
						locAxisSplit[1] = sqrtSequence - i;
						i = sqrtSequence;
						successSplit = 1;
					}
				}
			}
			
			if (successSplit == 0)
				numPasses = 3;
		}
	}
	if (numPasses == 3) {
		if (isPowOf2) {
			uint64_t maxPow8Strided = (uint64_t)pow(8, ((uint64_t)log2(maxSingleSizeStrided)) / 3);
			uint64_t maxSingleSizeStrided128 = app->configuration.sharedMemorySize / (128);
			uint64_t maxPow8_128 = (uint64_t)pow(8, ((uint64_t)log2(maxSingleSizeStrided128)) / 3);
				//unit stride
			if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxPow8_128 <= maxPow8Strided * maxSingleSizeStrided){
				locAxisSplit[0] = maxPow8_128;
			}
			if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[0] / maxPow8Strided <= maxSingleSizeStrided) {
				locAxisSplit[1] = maxPow8Strided;
				locAxisSplit[2] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / locAxisSplit[1] / locAxisSplit[0];
			}
			if (locAxisSplit[2] < 64) {
				locAxisSplit[1] = (locAxisSplit[2] == 0) ? locAxisSplit[1] / (64) : locAxisSplit[1] / (64 / locAxisSplit[2]);
				locAxisSplit[2] = 64;
			}
			if (locAxisSplit[2] > locAxisSplit[1]) {
				uint64_t swap = locAxisSplit[1];
				locAxisSplit[1] = locAxisSplit[2];
				locAxisSplit[2] = swap;
			}
		}
		else {
			uint64_t successSplit = 0;
			uint64_t sqrt3Sequence = (uint64_t)ceil(pow(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id], 1.0 / 3.0));
				for (uint64_t i = 0; i < sqrt3Sequence; i++) {
					if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (sqrt3Sequence - i) == 0) {
						uint64_t sqrt2Sequence = (uint64_t)ceil(sqrt(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrt3Sequence - i)));
						for (uint64_t j = 0; j < sqrt2Sequence; j++) {
							if ((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrt3Sequence - i)) % (sqrt2Sequence - j) == 0) {
								if ((sqrt3Sequence - i <= maxSingleSizeStrided) && (sqrt2Sequence - j <= maxSingleSizeStrided) && (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrt3Sequence - i) / (sqrt2Sequence - j) <= maxSingleSizeStridedHalfBandwidth)) {
									locAxisSplit[0] = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (sqrt3Sequence - i) / (sqrt2Sequence - j);
									locAxisSplit[1] = sqrt3Sequence - i;
									locAxisSplit[2] = sqrt2Sequence - j;
									i = sqrt3Sequence;
									j = sqrt2Sequence;
									successSplit = 1;
								}
							}
						}
					}
				}
			
			if (successSplit == 0)
				numPasses = 4;
		}
	}
	if (numPasses > 3) {
		//printf("sequence length exceeds boundaries\n");
		return FFT_ERROR_UNSUPPORTED_FFT_LENGTH;
	}

	for (uint64_t i = 0; i < numPasses; i++) {
		if ((locAxisSplit[0] % 2 != 0) && (locAxisSplit[i] % 2 == 0)) {
			uint64_t swap = locAxisSplit[0];
			locAxisSplit[0] = locAxisSplit[i];
			locAxisSplit[i] = swap;
		}
	}
	for (uint64_t i = 0; i < numPasses; i++) {
		if ((locAxisSplit[0] % 4 != 0) && (locAxisSplit[i] % 4 == 0)) {
			uint64_t swap = locAxisSplit[0];
			locAxisSplit[0] = locAxisSplit[i];
			locAxisSplit[i] = swap;
		}
	}
	for (uint64_t i = 0; i < numPasses; i++) {
		if ((locAxisSplit[0] % 8 != 0) && (locAxisSplit[i] % 8 == 0)) {
			uint64_t swap = locAxisSplit[0];
			locAxisSplit[0] = locAxisSplit[i];
			locAxisSplit[i] = swap;
		}
	}
	FFTPlan->numAxisUploads[axis_id] = numPasses;
	for (uint64_t k = 0; k < numPasses; k++) {
		tempSequence = locAxisSplit[k];
		uint64_t loc_multipliers[20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };//split the smaller sequence
		for (uint64_t i = 2; i < 14; i++) {
			if (tempSequence % i == 0) {
				tempSequence /= i;
				loc_multipliers[i]++;
				i--;
			}
		}
		uint64_t threadRegister = 8;
		uint64_t threadRadixRegister[14] = { 0 };
		uint64_t threadRegisterMin = 8;
		if (loc_multipliers[2] > 0) {
			if (loc_multipliers[3] > 0) {
				if (loc_multipliers[5] > 0) {
					if (loc_multipliers[7] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									threadRegister = 15;
									threadRadixRegister[2] = 14;
									threadRadixRegister[3] = 15;
									break;
								case 2:
									threadRegister = 15;
									threadRadixRegister[2] = 12;
									threadRadixRegister[3] = 12;
									break;
								case 3:
									threadRegister = 15;
									threadRadixRegister[2] = 12;
									threadRadixRegister[3] = 12;
									break;
								default:
									threadRegister = 16;
									threadRadixRegister[2] = 16;
									threadRadixRegister[3] = 12;
									break;
								}
								threadRadixRegister[5] = 15;
								threadRadixRegister[7] = 14;
								threadRadixRegister[11] = 0;
								threadRadixRegister[13] = 0;
								threadRegisterMin = 14;

					}
					else {

								switch (loc_multipliers[2]) {
								case 1:
									threadRegister = 6;
									threadRadixRegister[2] = 6;
									threadRadixRegister[3] = 6;
									threadRadixRegister[5] = 5;
									threadRegisterMin = 5;
									break;
								case 2:
									threadRegister = 12;
									threadRadixRegister[2] = 12;
									threadRadixRegister[3] = 12;
									threadRadixRegister[5] = 10;
									threadRegisterMin = 10;
									break;
								default:
									threadRegister = 12;
									threadRadixRegister[2] = 12;
									threadRadixRegister[3] = 12;
									threadRadixRegister[5] = 10;
									threadRegisterMin = 10;
									break;
								}
								threadRadixRegister[7] = 0;
								threadRadixRegister[11] = 0;
								threadRadixRegister[13] = 0;

					}
				}
				else
				{
					if (loc_multipliers[7] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									threadRegister = 7;
									threadRadixRegister[2] = 6;
									threadRadixRegister[3] = 6;
									threadRadixRegister[5] = 0;
									threadRadixRegister[7] = 7;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 6;
									break;
								case 2:
									threadRegister = 7;
									threadRadixRegister[2] = 6;
									threadRadixRegister[3] = 6;
									threadRadixRegister[5] = 0;
									threadRadixRegister[7] = 7;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 6;
									break;
								default:
									threadRegister = 8;
									threadRadixRegister[2] = 8;
									threadRadixRegister[3] = 6;
									threadRadixRegister[5] = 0;
									threadRadixRegister[7] = 7;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 6;
									break;

						}
					}
					else {
								switch (loc_multipliers[2]) {
								case 1:
									threadRegister = 6;
									threadRadixRegister[2] = 6;
									threadRadixRegister[3] = 6;
									threadRadixRegister[5] = 0;
									threadRadixRegister[7] = 0;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 6;
									break;
								case 2:
									threadRegister = 12;
									threadRadixRegister[2] = 12;
									threadRadixRegister[3] = 12;
									threadRadixRegister[5] = 0;
									threadRadixRegister[7] = 0;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 12;
									break;
								default:
									threadRegister = 12;
									threadRadixRegister[2] = 12;
									threadRadixRegister[3] = 12;
									threadRadixRegister[5] = 0;
									threadRadixRegister[7] = 0;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 12;
									break;
								}

					}
				}
			}
			else {
				if (loc_multipliers[5] > 0) {
					if (loc_multipliers[7] > 0) {


								switch (loc_multipliers[2]) {
								case 1:
									threadRegister = 10;
									threadRadixRegister[2] = 10;
									threadRadixRegister[3] = 0;
									threadRadixRegister[5] = 10;
									threadRadixRegister[7] = 7;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 7;
									break;
								case 2:
									threadRegister = 10;
									threadRadixRegister[2] = 10;
									threadRadixRegister[3] = 0;
									threadRadixRegister[5] = 10;
									threadRadixRegister[7] = 7;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 7;
									break;
								default:
									threadRegister = 10;
									threadRadixRegister[2] = 8;
									threadRadixRegister[3] = 0;
									threadRadixRegister[5] = 10;
									threadRadixRegister[7] = 7;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 7;
									break;
								}
							
						
					}
					else {
								switch (loc_multipliers[2]) {
								case 1:
									threadRegister = 10;
									threadRadixRegister[2] = 10;
									threadRadixRegister[3] = 0;
									threadRadixRegister[5] = 10;
									threadRadixRegister[7] = 0;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 10;
									break;
								case 2:
									threadRegister = 10;
									threadRadixRegister[2] = 10;
									threadRadixRegister[3] = 0;
									threadRadixRegister[5] = 10;
									threadRadixRegister[7] = 0;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 10;
									break;
								default:
									threadRegister = 10;
									threadRadixRegister[2] = 10;
									threadRadixRegister[3] = 0;
									threadRadixRegister[5] = 10;
									threadRadixRegister[7] = 0;
									threadRadixRegister[11] = 0;
									threadRadixRegister[13] = 0;
									threadRegisterMin = 10;
									break;
								}

					}
				}
				else
				{
					if (loc_multipliers[7] > 0) {
						switch (loc_multipliers[2]) {
						case 1:
							threadRegister = 14;
							threadRadixRegister[2] = 14;
							threadRadixRegister[3] = 0;
							threadRadixRegister[5] = 0;
							threadRadixRegister[7] = 14;
							threadRadixRegister[11] = 0;
							threadRadixRegister[13] = 0;
							threadRegisterMin = 14;
							break;
						case 2:
							threadRegister = 14;
							threadRadixRegister[2] = 14;
							threadRadixRegister[3] = 0;
							threadRadixRegister[5] = 0;
							threadRadixRegister[7] = 14;
							threadRadixRegister[11] = 0;
							threadRadixRegister[13] = 0;
							threadRegisterMin = 14;
							break;
						case 3:
							threadRegister = 14;
							threadRadixRegister[2] = 14;
							threadRadixRegister[3] = 0;
							threadRadixRegister[5] = 0;
							threadRadixRegister[7] = 14;
							threadRadixRegister[11] = 0;
							threadRadixRegister[13] = 0;
							threadRegisterMin = 14;
							break;
						default:
							threadRegister = 14;
							threadRadixRegister[2] = 14;
							threadRadixRegister[3] = 0;
							threadRadixRegister[5] = 0;
							threadRadixRegister[7] = 14;
							threadRadixRegister[11] = 0;
							threadRadixRegister[13] = 0;
							threadRegisterMin = 14;
							break;
						}
							
						
					}
					else {
						threadRegister = (loc_multipliers[2] > 2) ? 8 : (uint64_t)pow(2, loc_multipliers[2]);
						threadRadixRegister[2] = (loc_multipliers[2] > 2) ? 8 : (uint64_t)pow(2, loc_multipliers[2]);
						threadRadixRegister[3] = 0;
						threadRadixRegister[5] = 0;
						threadRadixRegister[7] = 0;
						threadRadixRegister[11] = 0;
						threadRadixRegister[13] = 0;
						threadRegisterMin = (loc_multipliers[2] > 2) ? 8 : (uint64_t)pow(2, loc_multipliers[2]);

					}
				}
			}
		}
		else {
			if (loc_multipliers[3] > 0) {
				if (loc_multipliers[5] > 0) {
					if (loc_multipliers[7] > 0) {
						threadRegister = 21;
						threadRadixRegister[2] = 0;
						threadRadixRegister[3] = 15;
						threadRadixRegister[5] = 15;
						threadRadixRegister[7] = 21;
						threadRadixRegister[11] = 0;
						threadRadixRegister[13] = 0;
						threadRegisterMin = 15;

					}
					else {
						threadRegister = 15;
						threadRadixRegister[2] = 0;
						threadRadixRegister[3] = 15;
						threadRadixRegister[5] = 15;
						threadRadixRegister[7] = 0;
						threadRadixRegister[11] = 0;
						threadRadixRegister[13] = 0;
						threadRegisterMin = 15;

					}
				}
				else
				{
					if (loc_multipliers[7] > 0) {
						if (loc_multipliers[3] == 1) {
							threadRegister = 21;
							threadRadixRegister[2] = 0;
							threadRadixRegister[3] = 21;
							threadRadixRegister[5] = 0;
							threadRadixRegister[7] = 21;
							threadRadixRegister[11] = 0;
							threadRadixRegister[13] = 0;
							threadRegisterMin = 21;

						}
						else {
							threadRegister = 9;
							threadRadixRegister[2] = 0;
							threadRadixRegister[3] = 9;
							threadRadixRegister[5] = 0;
							threadRadixRegister[7] = 7;
							threadRadixRegister[11] = 0;
							threadRadixRegister[13] = 0;
							threadRegisterMin = 7;
						}
					}
					else {
						if (loc_multipliers[3] == 1) {
							threadRegister = 3;
							threadRadixRegister[2] = 0;
							threadRadixRegister[3] = 3;
							threadRadixRegister[5] = 0;
							threadRadixRegister[7] = 0;
							threadRadixRegister[11] = 0;
							threadRadixRegister[13] = 0;
							threadRegisterMin = 3;

						}
						else {
							threadRegister = 9;
							threadRadixRegister[2] = 0;
							threadRadixRegister[3] = 9;
							threadRadixRegister[5] = 0;
							threadRadixRegister[7] = 0;
							threadRadixRegister[11] = 0;
							threadRadixRegister[13] = 0;
							threadRegisterMin = 9;

						}
					}
				}
			}
			else {
				if (loc_multipliers[5] > 0) {
					if (loc_multipliers[7] > 0) {
						threadRegister = 7;
						threadRadixRegister[2] = 0;
						threadRadixRegister[3] = 0;
						threadRadixRegister[5] = 5;
						threadRadixRegister[7] = 7;
						threadRadixRegister[11] = 0;
						threadRadixRegister[13] = 0;
						threadRegisterMin = 5;
					}
					else {
						threadRegister = 5;
						threadRadixRegister[2] = 0;
						threadRadixRegister[3] = 0;
						threadRadixRegister[5] = 5;
						threadRadixRegister[7] = 0;
						threadRadixRegister[11] = 0;
						threadRadixRegister[13] = 0;
						threadRegisterMin = 5;

					}
				}
				else
				{
					if (loc_multipliers[7] > 0) {
						threadRegister = 7;
						threadRadixRegister[2] = 0;
						threadRadixRegister[3] = 0;
						threadRadixRegister[5] = 0;
						threadRadixRegister[7] = 7;
						threadRadixRegister[11] = 0;
						threadRadixRegister[13] = 0;
						threadRegisterMin = 7;

					}
					else {
						return FFT_ERROR_UNSUPPORTED_RADIX;
					}
				}
			}

		}
		threadRadixRegister[8] = threadRadixRegister[2];
		threadRadixRegister[4] = threadRadixRegister[2];

		if (threadRadixRegister[8] % 8 == 0) {
			loc_multipliers[8] = loc_multipliers[2] / 3;
			loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[8] * 3;
		}
		if (threadRadixRegister[4] % 4 == 0) {
			loc_multipliers[4] = loc_multipliers[2] / 2;
			loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[4] * 2;
		}

		uint64_t maxBatchCoalesced = ((axis_id == 0) && (((k == 0) && 0) || (numPasses == 1))) ? 1 : app->configuration.coalescedMemory / complexSize;
		
		uint64_t j = 0;
		axes[k].layout.regAd = regAd;
		axes[k].layout.threadRegister = threadRegister;
		axes[k].layout.threadRegisterMin = threadRegisterMin;
		for (uint64_t i = 2; i < 14; i++) {
			axes[k].layout.threadRadixRegister[i] = threadRadixRegister[i];
		}
		axes[k].layout.numStages = 0;
		axes[k].layout.dim = locAxisSplit[k];
		uint64_t tempregAd = regAd;
		uint64_t switchregAd = 0;
		if (tempregAd > 1) {
			if (loc_multipliers[tempregAd] > 0) {
				loc_multipliers[tempregAd]--;
				switchregAd = tempregAd;
			}
			else {
				for (uint64_t i = 14; i > 1; i--) {
					if (loc_multipliers[i] > 0) {
						loc_multipliers[i]--;
						switchregAd = i;
						i = 1;
					}
				}
			}
		}
		for (uint64_t i = 14; i > 1; i--) {
			if (loc_multipliers[i] > 0) {
				axes[k].layout.stageRadix[j] = i;
				loc_multipliers[i]--;
				i++;
				j++;
				axes[k].layout.numStages++;
			}
		}
		if (switchregAd > 0) {
			axes[k].layout.stageRadix[axes[k].layout.numStages] = switchregAd;
			axes[k].layout.numStages++;
		}
		else {
			if (threadRegisterMin != threadRegister) {
				for (uint64_t i = 0; i < axes[k].layout.numStages; i++) {
					if (axes[k].layout.threadRadixRegister[axes[k].layout.stageRadix[i]] == threadRegisterMin) {
						j = axes[k].layout.stageRadix[i];
						axes[k].layout.stageRadix[i] = axes[k].layout.stageRadix[0];
						axes[k].layout.stageRadix[0] = j;
						i = axes[k].layout.numStages;
					}
				}
			}
		}
	}
	return FFT_SUCCESS;
}



void deleteAxis(FFTApplication* app, FFTAxis* axis) {


	hipError_t res = hipSuccess;
	if ((app->configuration.useLUT) && (!axis->referenceLUT) && (axis->bufferLUT != 0)) {
		res = hipFree(axis->bufferLUT);
		axis->bufferLUT = 0;
	}
	if (axis->FFTModule != 0) {
		res = hipModuleUnload(axis->FFTModule);
		axis->FFTModule = 0;
	}


}
void deleteFFT(FFTApplication* app) {

	hipError_t res_t = hipSuccess;

	if (!app->configuration.userTempBuffer) {
		if (app->configuration.allocateTempBuffer) {
			app->configuration.allocateTempBuffer = 0;

			if (app->configuration.tempBuffer[0] != 0) {
				res_t = hipFree(app->configuration.tempBuffer[0]);
				app->configuration.tempBuffer[0] = 0;
			}

			if (app->configuration.tempBuffer != 0) {
				free(app->configuration.tempBuffer);
				app->configuration.tempBuffer = 0;
			}
		}
		if (app->configuration.tempBufferSize != 0) {
			free(app->configuration.tempBufferSize);
			app->configuration.tempBufferSize = 0;
		}
	}
	if (!app->configuration.makeInversePlanOnly) {
		if (app->localFFTPlan != 0) {
			for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
				if (app->localFFTPlan->numAxisUploads[i] > 0) {
					for (uint64_t j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++)
						deleteAxis(app, &app->localFFTPlan->axes[i][j]);
				}
			}
			if (app->localFFTPlan != 0) {
				free(app->localFFTPlan);
				app->localFFTPlan = 0;
			}
		}
	}
	if (!app->configuration.makeForwardPlanOnly) {
		if (app->localFFTPlan_inverse != 0) {
			for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
				if (app->localFFTPlan_inverse->numAxisUploads[i] > 0) {
					for (uint64_t j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++)
						deleteAxis(app, &app->localFFTPlan_inverse->axes[i][j]);
				}
			}
			if (app->localFFTPlan_inverse != 0) {
				free(app->localFFTPlan_inverse);
				app->localFFTPlan_inverse = 0;
			}
		}
	}
}
void freeShaderGenFFT(FFTLayout* lt) {
	if (lt->disableThreadsStart) {
		free(lt->disableThreadsStart);
		lt->disableThreadsStart = 0;
	}
	if (lt->disableThreadsStart) {
		free(lt->disableThreadsEnd);
		lt->disableThreadsEnd = 0;
	}
	if (lt->regIDs) {
		for (uint64_t i = 0; i < lt->threadRegister * lt->regAd; i++) {
			if (lt->regIDs[i]) {
				free(lt->regIDs[i]);
				lt->regIDs[i] = 0;
			}
		}
		free(lt->regIDs);
		lt->regIDs = 0;
	}
}

FFTResult shaderGenFFT(char* output, FFTLayout* lt, const char* floatType, const char* floatTypeInputMemory, const char* floatTypeOutputMemory, const char* floatTypeKernelMemory, const char* uintType, uint64_t type) {
	FFTResult res = FFT_SUCCESS;
	//appendLicense(output);
	lt->output = output;
	lt->tempStr = (char*)malloc(sizeof(char) * lt->maxTempLength);
	if (!lt->tempStr) return FFT_ERROR_MALLOC_FAILED;
	lt->tempLen = 0;
	lt->currentLen = 0;
	char vecType[30];
	char vecTypeInput[30];
	char vecTypeOutput[30];


	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");

	if (!strcmp(floatTypeInputMemory, "double")) sprintf(vecTypeInput, "double2");

	if (!strcmp(floatTypeOutputMemory, "double")) sprintf(vecTypeOutput, "double2");
	sprintf(lt->gl_LocalInvocationID_x, "threadIdx.x");
	sprintf(lt->gl_LocalInvocationID_y, "threadIdx.y");
	sprintf(lt->gl_LocalInvocationID_z, "threadIdx.z");
	sprintf(lt->gl_GlobalInvocationID_x, "(threadIdx.x + blockIdx.x * blockDim.x)");
	sprintf(lt->gl_GlobalInvocationID_y, "(threadIdx.y + blockIdx.y * blockDim.y)");
	sprintf(lt->gl_GlobalInvocationID_z, "(threadIdx.z + blockIdx.z * blockDim.z)");
	sprintf(lt->gl_WorkGroupID_x, "blockIdx.x");
	sprintf(lt->gl_WorkGroupID_y, "blockIdx.y");
	sprintf(lt->gl_WorkGroupID_z, "blockIdx.z");
	sprintf(lt->gl_WorkGroupSize_x, "blockDim.x");
	sprintf(lt->gl_WorkGroupSize_y, "blockDim.y");
	sprintf(lt->gl_WorkGroupSize_z, "blockDim.z");


	sprintf(lt->stageInvocationID, "stageInvocationID");
	sprintf(lt->blockInvocationID, "blockInvocationID");
	sprintf(lt->tshuffle, "tshuffle");
	sprintf(lt->sharedStride, "sharedStride");
	sprintf(lt->combinedID, "combinedID");
	sprintf(lt->inoutID, "inoutID");
	sprintf(lt->sdataID, "sdataID");
	//sprintf(lt->tempReg, "temp");

	lt->disableThreadsStart = (char*)malloc(sizeof(char) * 500);
	if (!lt->disableThreadsStart) {
		freeShaderGenFFT(lt);
		return FFT_ERROR_MALLOC_FAILED;
	}
	lt->disableThreadsEnd = (char*)malloc(sizeof(char) * 2);
	if (!lt->disableThreadsEnd) {
		freeShaderGenFFT(lt);
		return FFT_ERROR_MALLOC_FAILED;
	}
	lt->disableThreadsStart[0] = 0;
	lt->disableThreadsEnd[0] = 0;



	// res = appendExtensions(lt, floatType, floatTypeInputMemory, floatTypeOutputMemory, floatTypeKernelMemory);
	// if (res != FFT_SUCCESS) {
	// 	freeShaderGenFFT(lt);
	// 	return res;
	// }


	res = appendConstantsFFT(lt, floatType, uintType);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}

	if ((!lt->LUT) && (!strcmp(floatType, "double"))) {
		res = appendSinCos20(lt, floatType, uintType);
		if (res != FFT_SUCCESS) {
			freeShaderGenFFT(lt);
			return res;
		}
	}

	if (strcmp(floatType, floatTypeInputMemory)) {
		res = appendConversion(lt, floatType, floatTypeInputMemory);
		if (res != FFT_SUCCESS) {
			freeShaderGenFFT(lt);
			return res;
		}
	}
	if (strcmp(floatType, floatTypeOutputMemory) && strcmp(floatTypeInputMemory, floatTypeOutputMemory)) {
		res = appendConversion(lt, floatType, floatTypeOutputMemory);
		if (res != FFT_SUCCESS) {
			freeShaderGenFFT(lt);
			return res;
		}
	}
	res = appendPushConstantsFFT(lt, floatType, uintType);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}
	uint64_t id = 0;
	res = appendInputLayoutFFT(lt, id, floatTypeInputMemory, type);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}
	id++;
	res = appendOutputLayoutFFT(lt, id, floatTypeOutputMemory, type);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}
	id++;

	if (lt->LUT) {
		res = appendLUTLayoutFFT(lt, id, floatType);
		if (res != FFT_SUCCESS) {
			freeShaderGenFFT(lt);
			return res;
		}
		id++;
	}

	uint64_t locType = (((type == 0) || (type == 5) || (type == 6) || (type == 120) || (type == 130) || (type == 140) || (type == 142)) && (lt->axisSwapped)) ? 1 : type;

	lt->tempLen = sprintf(lt->tempStr, "extern __shared__ float shared[];\n");
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}
	lt->tempLen = sprintf(lt->tempStr, "extern \"C\" __launch_bounds__(%" PRIu64 ") __global__ void FFT_main ", lt->localSize[0] * lt->localSize[1] * lt->localSize[2]);
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}


	lt->tempLen = sprintf(lt->tempStr, "(%s* inputs, %s* outputs", vecTypeInput, vecTypeOutput);
	
	
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}

	if (lt->LUT) {
		lt->tempLen = sprintf(lt->tempStr, ", %s* twiddleLUT", vecType);
		res = AppendLine(lt);
		if (res != FFT_SUCCESS) {
			freeShaderGenFFT(lt);
			return res;
		}
	}
	lt->tempLen = sprintf(lt->tempStr, ") {\n");
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}

	//lt->tempLen = sprintf(lt->tempStr, ", const PushConsts consts) {\n");
	res = appendSharedMemoryFFT(lt, floatType, uintType, locType);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}


	//if (type==0) lt->tempLen = sprintf(lt->tempStr, "return;\n");
	res = appendInitialization(lt, floatType, uintType, type);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}


	res = appendReadDataFFT(lt, floatType, floatTypeInputMemory, uintType, type);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}
	res = appendReorder4StepRead(lt, floatType, uintType, locType);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}

	res = threadDataOrder(lt, floatType, uintType, locType, 1);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}


	uint64_t stageSize = 1;
	uint64_t stageSizeSum = 0;
	double PI_const = 3.1415926535897932384626433832795;
	double stageAngle = (lt->inverse) ? PI_const : -PI_const;
	for (uint64_t i = 0; i < lt->numStages; i++) {
		if ((i == lt->numStages - 1) && (lt->regAd > 1)) {
			res = appendRadixStage(lt, floatType, uintType, stageSize, stageSizeSum, stageAngle, lt->stageRadix[i], locType);
			if (res != FFT_SUCCESS) {
				freeShaderGenFFT(lt);
				return res;
			}
		}
		else {

			res = appendRadixStage(lt, floatType, uintType, stageSize, stageSizeSum, stageAngle, lt->stageRadix[i], locType);
			if (res != FFT_SUCCESS) {
				freeShaderGenFFT(lt);
				return res;
			}
			switch (lt->stageRadix[i]) {
			case 2:
				stageSizeSum += stageSize;
				break;
			case 3:
				stageSizeSum += stageSize * 2;
				break;
			case 4:
				stageSizeSum += stageSize * 2;
				break;
			case 5:
				stageSizeSum += stageSize * 4;
				break;
			case 7:
				stageSizeSum += stageSize * 6;
				break;
			case 8:
				stageSizeSum += stageSize * 3;
				break;
			case 11:
				stageSizeSum += stageSize * 10;
				break;
			case 13:
				stageSizeSum += stageSize * 12;
				break;
			}
			if (i == lt->numStages - 1) {
				res = appendRadixShuffle(lt, floatType, uintType, stageSize, stageSizeSum, stageAngle, lt->stageRadix[i], lt->stageRadix[i], locType);
				if (res != FFT_SUCCESS) {
					freeShaderGenFFT(lt);
					return res;
				}
			}
			else {
				res = appendRadixShuffle(lt, floatType, uintType, stageSize, stageSizeSum, stageAngle, lt->stageRadix[i], lt->stageRadix[i + 1], locType);
				if (res != FFT_SUCCESS) {
					freeShaderGenFFT(lt);
					return res;
				}
			}
			stageSize *= lt->stageRadix[i];
			stageAngle /= lt->stageRadix[i];
		}
	}


	res = threadDataOrder(lt, floatType, uintType, locType, 0);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}
	res = appendReorder4StepWrite(lt, floatType, uintType, locType);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}

	res = appendWriteDataFFT(lt, floatType, floatTypeOutputMemory, uintType, type);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}

	lt->tempLen = sprintf(lt->tempStr, "}\n");
	res = AppendLine(lt);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(lt);
		return res;
	}
	freeShaderGenFFT(lt);


	// char fileName[30] = "../kernel/kernel_";
	// if(lt->size[2] != 1){
	// sprintf(fileName, "../kernel/kernel_%" PRIu64 "x%" PRIu64 "x%" PRIu64 ".h", lt->size[0], lt->size[1], lt->size[2]);
	// }else if(lt->size[1] != 1){
	// 	sprintf(fileName, "kernel_%" PRIu64 "x%" PRIu64 ".h", lt->size[0], lt->size[1]);
	// }else{
	// 	sprintf(fileName, "kernel_%" PRIu64 ".h", lt->size[0]);
	// }
	// FILE *fp=fopen(fileName,"a");
	// fprintf(fp, "%s",output);
	// fclose(fp);

	return res;
}


FFTResult FFTPlanAxis(FFTApplication* app, FFTPlan* FFTPlan, uint64_t axis_id, uint64_t axis_upload_id, uint64_t inverse) {
	//get radix stages
	FFTResult resFFT = FFT_SUCCESS;
	hipError_t res = hipSuccess;

	FFTAxis* axis = &FFTPlan->axes[axis_id][axis_upload_id];
	axis->layout.warpSize = app->configuration.warpSize;
	axis->layout.numSharedBanks = app->configuration.numSharedBanks;
	axis->layout.useUint64 = app->configuration.useUint64;
	uint64_t complexSize;
	if (app->configuration.doublePrecision) complexSize = (2 * sizeof(double));
	
	axis->layout.complexSize = complexSize;
	axis->layout.supportAxis = 0;

	uint64_t maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / complexSize;
	uint64_t maxSequenceLengthSharedMemoryPow2 = app->configuration.sharedMemorySizePow2 / complexSize;
	uint64_t maxSingleSizeStrided = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySize / complexSize;
	uint64_t maxSingleSizeStridedPow2 = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySizePow2 / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySizePow2 / complexSize;

	axis->layout.stageStartSize = 1;
	for (uint64_t i = 0; i < axis_upload_id; i++)
		axis->layout.stageStartSize *= FFTPlan->axisSplit[axis_id][i];


	axis->layout.firstStageStartSize = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / FFTPlan->axisSplit[axis_id][FFTPlan->numAxisUploads[axis_id] - 1];


	if (axis_id == 0) {
		//configure radix stages
		axis->layout.fft_dim_x = axis->layout.stageStartSize;
	}
	else {
		axis->layout.fft_dim_x = FFTPlan->actualFFTSizePerAxis[axis_id][0];
	}

	if ((axis_id == 0) && ((FFTPlan->numAxisUploads[axis_id] == 1) || ((axis_upload_id == 0) && (!app->configuration.reorderFourStep)))) {
		maxSequenceLengthSharedMemory *= axis->layout.regAd;
		maxSequenceLengthSharedMemoryPow2 = (uint64_t)pow(2, (uint64_t)log2(maxSequenceLengthSharedMemory));
	}
	else {
		maxSingleSizeStrided *= axis->layout.regAd;
		maxSingleSizeStridedPow2 = (uint64_t)pow(2, (uint64_t)log2(maxSingleSizeStrided));
	}


	axis->layout.reorderFourStep = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.reorderFourStep : 0;
	//uint64_t passID = FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id;
	axis->layout.fft_dim_full = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
	if ((FFTPlan->numAxisUploads[axis_id] > 1) && (app->configuration.reorderFourStep) && (!app->configuration.userTempBuffer) && (app->configuration.allocateTempBuffer == 0)) {
		app->configuration.allocateTempBuffer = 1;

		app->configuration.tempBuffer = (void**)malloc(sizeof(void*));
		if (!app->configuration.tempBuffer) {
			deleteFFT(app);
			return FFT_ERROR_MALLOC_FAILED;
		}
		res = hipMalloc(app->configuration.tempBuffer, app->configuration.tempBufferSize[0]);
		if (res != hipSuccess) {
			deleteFFT(app);
			return FFT_ERROR_FAILED_TO_ALLOCATE;
		}
	}
	//allocate LUT
	if (app->configuration.useLUT) {
		double double_PI = 3.1415926535897932384626433832795;
		uint64_t dimMult = 1;
		uint64_t maxStageSum = 0;
		for (uint64_t i = 0; i < axis->layout.numStages; i++) {
			switch (axis->layout.stageRadix[i]) {
			case 2:
				maxStageSum += dimMult;
				break;
			case 3:
				maxStageSum += dimMult * 2;
				break;
			case 4:
				maxStageSum += dimMult * 2;
				break;
			case 5:
				maxStageSum += dimMult * 4;
				break;
			case 7:
				maxStageSum += dimMult * 6;
				break;
			case 8:
				maxStageSum += dimMult * 3;
				break;
			case 11:
				maxStageSum += dimMult * 10;
				break;
			case 13:
				maxStageSum += dimMult * 12;
				break;
			}
			dimMult *= axis->layout.stageRadix[i];
		}
		axis->layout.maxStageSumLUT = maxStageSum;
		dimMult = 1;
		if (app->configuration.doublePrecision) {
            if (axis_upload_id > 0) {
				axis->bufferLUTSize = (maxStageSum + axis->layout.stageStartSize * axis->layout.dim) * 2 * sizeof(double);
			}
			else {
                axis->bufferLUTSize = (maxStageSum) * 2 * sizeof(double);
			}

			double* tempLUT = (double*)malloc(axis->bufferLUTSize);
			if (!tempLUT) {
				deleteFFT(app);
				return FFT_ERROR_MALLOC_FAILED;
			}
			uint64_t localStageSize = 1;
			uint64_t localStageSum = 0;

			// for (uint64_t i = 0; i < axis->layout.numStages; i++) {
			// 	std::cout<<"stageradix = "<<axis->layout.stageRadix[i]<<"    ";
			// }
			// std::cout<<"\n";

			
			for (uint64_t i = 0; i < axis->layout.numStages; i++) {
				if ((axis->layout.stageRadix[i] & (axis->layout.stageRadix[i] - 1)) == 0) {
					for (uint64_t k = 0; k < log2(axis->layout.stageRadix[i]); k++) {
						for (uint64_t j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = cos(j * double_PI / localStageSize / pow(2, k));
							tempLUT[2 * (j + localStageSum) + 1] = sin(j * double_PI / localStageSize / pow(2, k));
						}
						localStageSum += localStageSize;
					}
					localStageSize *= axis->layout.stageRadix[i];
				}
				else {
					for (uint64_t k = (axis->layout.stageRadix[i] - 1); k > 0; k--) {
						for (uint64_t j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = cos(j * 2.0 * k / axis->layout.stageRadix[i] * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = sin(j * 2.0 * k / axis->layout.stageRadix[i] * double_PI / localStageSize);
						}
						localStageSum += localStageSize;
					}
					localStageSize *= axis->layout.stageRadix[i];
				}
			}				


			if (axis_upload_id > 0) {
				
				for (uint64_t i = 0; i < axis->layout.stageStartSize; i++) {
					for (uint64_t j = 0; j < axis->layout.dim; j++) {
						double angle = 2 * double_PI * ((i * j) / (double)(axis->layout.stageStartSize * axis->layout.dim));
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->layout.stageStartSize)] = cos(angle);
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->layout.stageStartSize) + 1] = sin(angle);
					}
				}
			}
			
			axis->referenceLUT = 0;
			if ((!inverse) && (!app->configuration.makeForwardPlanOnly)) {
				
				axis->bufferLUT = app->localFFTPlan_inverse->axes[axis_id][axis_upload_id].bufferLUT;

				axis->bufferLUTSize = app->localFFTPlan_inverse->axes[axis_id][axis_upload_id].bufferLUTSize;
				axis->referenceLUT = 1;

			}
			else {
				if (((axis_id == 1) || (axis_id == 2)) && (!((!app->configuration.reorderFourStep) && (FFTPlan->numAxisUploads[axis_id] > 1))) && ((axis->layout.fft_dim_full == FFTPlan->axes[0][0].layout.fft_dim_full) && (FFTPlan->numAxisUploads[axis_id] == 1) && (axis->layout.fft_dim_full < maxSingleSizeStrided / axis->layout.regAd))) {
					axis->bufferLUT = FFTPlan->axes[0][axis_upload_id].bufferLUT;

					axis->bufferLUTSize = FFTPlan->axes[0][axis_upload_id].bufferLUTSize;
					axis->referenceLUT = 1;
				}
				else {
					if ((axis_id == 2) && (axis->layout.fft_dim_full == FFTPlan->axes[1][0].layout.fft_dim_full)) {
						axis->bufferLUT = FFTPlan->axes[1][axis_upload_id].bufferLUT;
						axis->bufferLUTSize = FFTPlan->axes[1][axis_upload_id].bufferLUTSize;
						axis->referenceLUT = 1;
					}
					else {
						res = hipMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
						if (res != hipSuccess) {
							deleteFFT(app);
							free(tempLUT);
							tempLUT = 0;
							return FFT_ERROR_FAILED_TO_ALLOCATE;
						}
					
						res = hipMemcpy(axis->bufferLUT, tempLUT, axis->bufferLUTSize, hipMemcpyHostToDevice);
						if (res != hipSuccess) {
							deleteFFT(app);
							free(tempLUT);
							tempLUT = 0;
							return FFT_ERROR_FAILED_TO_ALLOCATE;
						}
					}
				}
			}
			free(tempLUT);
			tempLUT = 0;
		}
	}

	//configure strides



	uint64_t* axisStride = axis->layout.inputStride;
	uint64_t* usedStride = app->configuration.bufferStride;
	// if ((!inverse) && (axis_id == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1)) usedStride = app->configuration.inputBufferStride;
	
    // if ((inverse) && (axis_id == app->configuration.FFTdim - 1) && (((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.reorderFourStep)) || ((axis_upload_id == 0) && (!app->configuration.reorderFourStep))) && (!app->configuration.inverseReturnToInputBuffer)) usedStride = app->configuration.inputBufferStride;


	if ((!inverse) && (axis_id == 0) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (0)) usedStride = app->configuration.inputBufferStride;
	if ((inverse) && (axis_id == app->configuration.FFTdim - 1) && (((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.reorderFourStep)) || ((axis_upload_id == 0) && (!app->configuration.reorderFourStep))) && (0) && (!app->configuration.inverseReturnToInputBuffer)) usedStride = app->configuration.inputBufferStride;


	axisStride[0] = 1;

	if (axis_id == 0) {
		axisStride[1] = usedStride[0];
		axisStride[2] = usedStride[1];
	}
	if (axis_id == 1)
	{
		axisStride[1] = usedStride[0];
		axisStride[2] = usedStride[1];
	}
	if (axis_id == 2)
	{
		axisStride[1] = usedStride[1];
		axisStride[2] = usedStride[0];
	}

	axisStride[3] = usedStride[2];
	axisStride[4] = axisStride[3];


	axisStride = axis->layout.outputStride;
	usedStride = app->configuration.bufferStride;

	if ((!inverse) && (axis_id == app->configuration.FFTdim - 1) && (axis_upload_id == 0) && (0)) usedStride = app->configuration.outputBufferStride;
	if ((inverse) && (axis_id == 0) && (((axis_upload_id == 0) && (app->configuration.reorderFourStep)) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (!app->configuration.reorderFourStep))) && ((0))) {

		usedStride = app->configuration.outputBufferStride;}
	if ((inverse) && (axis_id == 0) && (((axis_upload_id == 0) && (0)) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (!app->configuration.reorderFourStep))) && (app->configuration.inverseReturnToInputBuffer)) {
		usedStride = app->configuration.inputBufferStride;
	}

	axisStride[0] = 1;

	if (axis_id == 0) {
		axisStride[1] = usedStride[0];
		axisStride[2] = usedStride[1];
	}
	if (axis_id == 1)
	{
		axisStride[1] = usedStride[0];
		axisStride[2] = usedStride[1];
	}
	if (axis_id == 2)
	{
		axisStride[1] = usedStride[1];
		axisStride[2] = usedStride[0];
	}

	axisStride[3] = usedStride[2];
	axisStride[4] = axisStride[3];






	axis->layout.actualInverse = inverse;
	axis->layout.inverse = inverse;
		
	

	axis->layout.inputOffset = 0;
	axis->layout.outputOffset = 0;

	uint64_t storageComplexSize;
	if (app->configuration.doublePrecision) storageComplexSize = (2 * sizeof(double));

	uint64_t initPageSize = -1;


	
	uint64_t totalSize = 0;
	uint64_t locPageSize = initPageSize;
	if ((axis->layout.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
		if (axis_upload_id > 0) {
			for (uint64_t i = 0; i < 1; i++) {
				totalSize += app->configuration.bufferSize[i];
				if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
			}
		}
		else {
			for (uint64_t i = 0; i < 1; i++) {
				totalSize += app->configuration.tempBufferSize[i];
                if (app->configuration.tempBufferSize[i] < locPageSize) locPageSize = app->configuration.tempBufferSize[i];

			}
		}
	else {
		for (uint64_t i = 0; i < 1; i++) {
			totalSize += app->configuration.bufferSize[i];
			if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];

		}
	}

	axis->layout.inputBufferBlockSize = (uint64_t)ceil(locPageSize / (double)storageComplexSize);
	axis->layout.inputBufferBlockNum = (uint64_t)ceil(totalSize / (double)(axis->layout.inputBufferBlockSize * storageComplexSize));
			//if (axis->layout.inputBufferBlockNum == 1) axis->layout.inputBufferBlockSize = totalSize / storageComplexSize;

		
	


	totalSize = 0;
	locPageSize = initPageSize;
	if ((axis->layout.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
		if (axis_upload_id == 1) {
			for (uint64_t i = 0; i < 1; i++) {
				totalSize += app->configuration.bufferSize[i];
				if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
			}
		}
		else {
			for (uint64_t i = 0; i < 1; i++) {
				totalSize += app->configuration.tempBufferSize[i];
				if (app->configuration.tempBufferSize[i] < locPageSize) locPageSize = app->configuration.tempBufferSize[i];
			}
		}
	else {
		for (uint64_t i = 0; i < 1; i++) {
			totalSize += app->configuration.bufferSize[i];
			if (app->configuration.bufferSize[i] < locPageSize) locPageSize = app->configuration.bufferSize[i];
		}
	}
	axis->layout.outputBufferBlockSize = (uint64_t)ceil(locPageSize / (double)storageComplexSize);
	axis->layout.outputBufferBlockNum = (uint64_t)ceil(totalSize / (double)(axis->layout.outputBufferBlockSize * storageComplexSize));
		//if (axis->layout.outputBufferBlockNum == 1) axis->layout.outputBufferBlockSize = totalSize / storageComplexSize;


	if (axis->layout.inputBufferBlockNum == 0) axis->layout.inputBufferBlockNum = 1;
	if (axis->layout.outputBufferBlockNum == 0) axis->layout.outputBufferBlockNum = 1;

	axis->numBindings = 2;
	axis->layout.numBuffersBound[0] = axis->layout.inputBufferBlockNum;
	axis->layout.numBuffersBound[1] = axis->layout.outputBufferBlockNum;
	axis->layout.numBuffersBound[2] = 0;
	axis->layout.numBuffersBound[3] = 0;

	if (app->configuration.useLUT) {
		axis->layout.numBuffersBound[axis->numBindings] = 1;
		axis->numBindings++;
	}

	resFFT = FFTCheckBuffer(app, axis, 1, 0);
	if (resFFT != FFT_SUCCESS) {
		deleteFFT(app);
		return resFFT;
	}
	resFFT = FFTUpdateBuffer(app, FFTPlan, axis, axis_id, axis_upload_id, inverse);
	if (resFFT != FFT_SUCCESS) {
		deleteFFT(app);
		return resFFT;
	}
	{

		uint64_t maxBatchCoalesced = app->configuration.coalescedMemory / complexSize;
		axis->groupedBatch = maxBatchCoalesced;
		

		if (((FFTPlan->numAxisUploads[axis_id] == 1) && (axis_id == 0)) || ((axis_id == 0) && (!app->configuration.reorderFourStep) && (axis_upload_id == 0))) {
			axis->groupedBatch = (maxSequenceLengthSharedMemoryPow2 / axis->layout.dim > axis->groupedBatch) ? maxSequenceLengthSharedMemoryPow2 / axis->layout.dim : axis->groupedBatch;
		}
		else {
			axis->groupedBatch = (maxSingleSizeStridedPow2 / axis->layout.dim > 1) ? maxSingleSizeStridedPow2 / axis->layout.dim * axis->groupedBatch : axis->groupedBatch;
		}
		

		if ((FFTPlan->numAxisUploads[axis_id] == 2) && (axis_upload_id == 0) && (axis->layout.dim * maxBatchCoalesced <= maxSequenceLengthSharedMemory)) {
			axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
		}

		if ((FFTPlan->numAxisUploads[axis_id] == 3) && (axis_upload_id == 0) && (axis->layout.dim < maxSequenceLengthSharedMemory / (2 * complexSize))) {
			axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
		}
		if (axis->groupedBatch < maxBatchCoalesced) axis->groupedBatch = maxBatchCoalesced;
		axis->groupedBatch = (axis->groupedBatch / maxBatchCoalesced) * maxBatchCoalesced;


		if (!((axis_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1)) && !((axis_id == 0) && (axis_upload_id == 0) && (!app->configuration.reorderFourStep)) && (axis->layout.dim > maxSingleSizeStrided)) {
			axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
		}

		if ((app->configuration.halfThreads) && (axis->groupedBatch * axis->layout.dim * complexSize >= app->configuration.sharedMemorySize))
			axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
		if (axis->groupedBatch > app->configuration.warpSize) axis->groupedBatch = (axis->groupedBatch / app->configuration.warpSize) * app->configuration.warpSize;
		if (axis->groupedBatch > 2 * maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / (2 * maxBatchCoalesced)) * (2 * maxBatchCoalesced);
		if (axis->groupedBatch > 4 * maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / (4 * maxBatchCoalesced)) * (2 * maxBatchCoalesced);
		uint64_t maxThreadNum = maxSequenceLengthSharedMemory / (axis->layout.threadRegisterMin * axis->layout.regAd);
		axis->layout.axisSwapped = 0;
		uint64_t r2cmult = (axis->layout.mergeSequencesR2C) ? 2 : 1;

		if (axis_id == 0) {
			if (axis_upload_id == 0) {

				axis->axisBlock[0] = (axis->layout.dim / axis->layout.threadRegisterMin / axis->layout.regAd > 1) ? axis->layout.dim / axis->layout.threadRegisterMin / axis->layout.regAd : 1;


				if (axis->axisBlock[0] > maxThreadNum) axis->axisBlock[0] = maxThreadNum;
				if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
				if (app->configuration.reorderFourStep && (FFTPlan->numAxisUploads[axis_id] > 1))
					axis->axisBlock[1] = axis->groupedBatch;
				else {
					//axis->axisBlock[1] = (axis->axisBlock[0] < app->configuration.warpSize) ? app->configuration.warpSize / axis->axisBlock[0] : 1;
					axis->axisBlock[1] = ((axis->axisBlock[0] < app->configuration.aimThreads)) ? app->configuration.aimThreads / axis->axisBlock[0] : 1;
				}
				uint64_t currentAxisBlock1 = axis->axisBlock[1];
				for (uint64_t i = currentAxisBlock1; i < 2 * currentAxisBlock1; i++) {
					if (((FFTPlan->numAxisUploads[0] > 1) && (((FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->layout.dim) % i) == 0)) || ((FFTPlan->numAxisUploads[0] == 1) && (((FFTPlan->actualFFTSizePerAxis[axis_id][1] / r2cmult) % i) == 0))) {
						if (i * axis->layout.dim * complexSize <= app->configuration.sharedMemorySize) axis->axisBlock[1] = i;
						i = 2 * currentAxisBlock1;
					}
				}

				if ((FFTPlan->numAxisUploads[0] > 1) && ((uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->layout.dim) < axis->axisBlock[1])) axis->axisBlock[1] = (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->layout.dim);
				if ((axis->layout.mergeSequencesR2C != 0) && (axis->layout.dim * axis->axisBlock[1] >= maxSequenceLengthSharedMemory)) {
					axis->layout.mergeSequencesR2C = 0;
					r2cmult = 1;
				}
				if ((FFTPlan->numAxisUploads[0] == 1) && ((uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)r2cmult) < axis->axisBlock[1])) axis->axisBlock[1] = (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)r2cmult);

				if (axis->axisBlock[1] > app->configuration.maxComputeWorkGroupSize[1]) axis->axisBlock[1] = app->configuration.maxComputeWorkGroupSize[1];
				if (axis->axisBlock[0] * axis->axisBlock[1] > app->configuration.maxThreadsNum) axis->axisBlock[1] /= 2;
				while ((axis->axisBlock[1] * (axis->layout.dim / axis->layout.regAd)) > maxSequenceLengthSharedMemory) axis->axisBlock[1] /= 2;
				if (((axis->layout.dim % 2 == 0) || (axis->axisBlock[0] < app->configuration.numSharedBanks / 4)) && (!((!app->configuration.reorderFourStep) && (FFTPlan->numAxisUploads[0] > 1))) && (axis->axisBlock[1] > 1) && (axis->axisBlock[1] * axis->layout.dim < maxSequenceLengthSharedMemoryPow2)) {


					uint64_t temp = axis->axisBlock[1];
					axis->axisBlock[1] = axis->axisBlock[0];
					axis->axisBlock[0] = temp;
					axis->layout.axisSwapped = 1;
				}
				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->layout.dim;

			}
			else {

				axis->axisBlock[1] = (axis->layout.dim / axis->layout.threadRegisterMin / axis->layout.regAd > 1) ? axis->layout.dim / axis->layout.threadRegisterMin / axis->layout.regAd : 1;
				uint64_t scale = app->configuration.aimThreads / axis->axisBlock[1] / axis->groupedBatch;
				if (scale > 1) axis->groupedBatch *= scale;
				axis->axisBlock[0] = (axis->layout.stageStartSize > axis->groupedBatch) ? axis->groupedBatch : axis->layout.stageStartSize;
				if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
				if (axis->axisBlock[0] * axis->axisBlock[1] > app->configuration.maxThreadsNum) {
					for (uint64_t i = 1; i <= axis->axisBlock[0]; i++) {
						if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= app->configuration.maxThreadsNum)
						{
							axis->axisBlock[0] /= i;
							i = axis->axisBlock[0] + 1;
						}

					}
				}
				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->layout.dim;
			}

		}
		if (axis_id == 1) {

			axis->axisBlock[1] = (axis->layout.dim / axis->layout.threadRegisterMin / axis->layout.regAd > 1) ? axis->layout.dim / axis->layout.threadRegisterMin / axis->layout.regAd : 1;

		
			axis->axisBlock[0] = (FFTPlan->actualFFTSizePerAxis[axis_id][0] > axis->groupedBatch) ? axis->groupedBatch : FFTPlan->actualFFTSizePerAxis[axis_id][0];
				
			
			if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
			if (axis->axisBlock[0] * axis->axisBlock[1] > app->configuration.maxThreadsNum) {
				for (uint64_t i = 1; i <= axis->axisBlock[0]; i++) {
					if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= app->configuration.maxThreadsNum)
					{
						axis->axisBlock[0] /= i;
						i = axis->axisBlock[0] + 1;
					}

				}
			}
			axis->axisBlock[2] = 1;
			axis->axisBlock[3] = axis->layout.dim;

		}
		if (axis_id == 2) {
			axis->axisBlock[1] = (axis->layout.dim / axis->layout.threadRegisterMin / axis->layout.regAd > 1) ? axis->layout.dim / axis->layout.threadRegisterMin / axis->layout.regAd : 1;


			axis->axisBlock[0] = (FFTPlan->actualFFTSizePerAxis[axis_id][0] > axis->groupedBatch) ? axis->groupedBatch : FFTPlan->actualFFTSizePerAxis[axis_id][0];
				
			
			if (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0]) axis->axisBlock[0] = app->configuration.maxComputeWorkGroupSize[0];
			if (axis->axisBlock[0] * axis->axisBlock[1] > app->configuration.maxThreadsNum) {
				for (uint64_t i = 1; i <= axis->axisBlock[0]; i++) {
					if ((axis->axisBlock[0] / i) * axis->axisBlock[1] <= app->configuration.maxThreadsNum)
					{
						axis->axisBlock[0] /= i;
						i = axis->axisBlock[0] + 1;
					}

				}
			}
			axis->axisBlock[2] = 1;
			axis->axisBlock[3] = axis->layout.dim;
		}



		uint64_t tempSize[3] = { FFTPlan->actualFFTSizePerAxis[axis_id][0], FFTPlan->actualFFTSizePerAxis[axis_id][1], FFTPlan->actualFFTSizePerAxis[axis_id][2] };


		if (axis_id == 0) {
			if (axis_upload_id == 0)
				tempSize[0] = FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->layout.dim / axis->axisBlock[1];
			else
				tempSize[0] = FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->layout.dim / axis->axisBlock[0];
			
			if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->layout.performWorkGroupShift[0] = 1;
			else  axis->layout.performWorkGroupShift[0] = 0;
			if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->layout.performWorkGroupShift[1] = 1;
			else  axis->layout.performWorkGroupShift[1] = 0;
			if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->layout.performWorkGroupShift[2] = 1;
			else  axis->layout.performWorkGroupShift[2] = 0;
		}
		if (axis_id == 1) {
			tempSize[0] = (0) ? (uint64_t)ceil((FFTPlan->actualFFTSizePerAxis[axis_id][0] / 2 + 1) / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)axis->layout.dim) : (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)axis->layout.dim);
			tempSize[1] = 1;
			tempSize[2] = FFTPlan->actualFFTSizePerAxis[axis_id][2];


			if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->layout.performWorkGroupShift[0] = 1;
			else  axis->layout.performWorkGroupShift[0] = 0;
			if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->layout.performWorkGroupShift[1] = 1;
			else  axis->layout.performWorkGroupShift[1] = 0;
			if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->layout.performWorkGroupShift[2] = 1;
			else  axis->layout.performWorkGroupShift[2] = 0;

		}
		if (axis_id == 2) {
			tempSize[0] = (0) ? (uint64_t)ceil((FFTPlan->actualFFTSizePerAxis[axis_id][0] / 2 + 1) / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][2] / (double)axis->layout.dim) : (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][2] / (double)axis->layout.dim);
			tempSize[1] = 1;
			tempSize[2] = FFTPlan->actualFFTSizePerAxis[axis_id][1];
			//if (app->configuration.actualPerformR2C == 1) tempSize[0] = (uint64_t)ceil(tempSize[0] / 2.0);

			if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->layout.performWorkGroupShift[0] = 1;
			else  axis->layout.performWorkGroupShift[0] = 0;
			if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->layout.performWorkGroupShift[1] = 1;
			else  axis->layout.performWorkGroupShift[1] = 0;
			if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->layout.performWorkGroupShift[2] = 1;
			else  axis->layout.performWorkGroupShift[2] = 0;

		}
		
		axis->layout.localSize[0] = axis->axisBlock[0];
		axis->layout.localSize[1] = axis->axisBlock[1];
		axis->layout.localSize[2] = axis->axisBlock[2];

		


		axis->layout.numBatches = 1;
		axis->layout.numKernels = 1;
		axis->layout.sharedMemSize = app->configuration.sharedMemorySize;
		axis->layout.sharedMemSizePow2 = app->configuration.sharedMemorySizePow2;
		axis->layout.normalize = app->configuration.normalize;
		axis->layout.size[0] = FFTPlan->actualFFTSizePerAxis[axis_id][0];
		axis->layout.size[1] = FFTPlan->actualFFTSizePerAxis[axis_id][1];
		axis->layout.size[2] = FFTPlan->actualFFTSizePerAxis[axis_id][2];
		axis->layout.axis_id = axis_id;
		axis->layout.axis_upload_id = axis_upload_id;

		char floatTypeInputMemory[10];
		char floatTypeOutputMemory[10];
		char floatTypeKernelMemory[10];
		char floatType[10];
		axis->layout.unroll = 1;
		axis->layout.LUT = app->configuration.useLUT;
		if (app->configuration.doublePrecision) {
			sprintf(floatType, "double");
			sprintf(floatTypeInputMemory, "double");
			sprintf(floatTypeOutputMemory, "double");
			sprintf(floatTypeKernelMemory, "double");
		}

		char uintType[20] = "";
		if (!app->configuration.useUint64) {
			sprintf(uintType, "unsigned int");
		}
		else {
			sprintf(uintType, "unsigned long long");
		}

		uint64_t type = 0;
		if ((axis_id == 0) && (axis_upload_id == 0)) type = 0;
		if (axis_id != 0) type = 1;
		if ((axis_id == 0) && (axis_upload_id > 0)) type = 2;

		

		axis->layout.maxCodeLength = app->configuration.maxCodeLength;
		axis->layout.maxTempLength = app->configuration.maxTempLength;
		char* code0 = (char*)malloc(sizeof(char) * app->configuration.maxCodeLength);
		if (!code0) {
			deleteFFT(app);
			return FFT_ERROR_MALLOC_FAILED;
		}
		shaderGenFFT(code0, &axis->layout, floatType, floatTypeInputMemory, floatTypeOutputMemory, floatTypeKernelMemory, uintType, type);

		hiprtcProgram prog;

		enum hiprtcResult result = hiprtcCreateProgram(&prog,         // prog
			code0,         // buffer
			"mainFFT.hip",    // name
			0,             // numHeaders
			0,          // headers
			0);        // includeNames
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcCreateProgram error: %s\n", hiprtcGetErrorString(result));
			free(code0);
			code0 = 0;
			deleteFFT(app);
			return FFT_ERROR_FAILED_TO_CREATE_PROGRAM;
		}

		result = hiprtcAddNameExpression(prog, "&consts");
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcAddNameExpression error: %s\n", hiprtcGetErrorString(result));
			free(code0);
			code0 = 0;
			deleteFFT(app);
			return FFT_ERROR_FAILED_TO_ADD_NAME_EXPRESSION;
		}


		result = hiprtcCompileProgram(prog,  // prog
			0,     // numOptions
			0); // options
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcCompileProgram error: %s\n", hiprtcGetErrorString(result));
			char* log = (char*)malloc(sizeof(char) * 100000);
			if (!log) {
				free(code0);
				code0 = 0;
				deleteFFT(app);
				return FFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
			}
			else {
				hiprtcGetProgramLog(prog, log);
				printf("%s\n", log);
				free(log);
				log = 0;
				printf("%s\n", code0);
				free(code0);
				code0 = 0;
				deleteFFT(app);
				return FFT_ERROR_FAILED_TO_COMPILE_PROGRAM;
			}
		}
		size_t codeSize;
		result = hiprtcGetCodeSize(prog, &codeSize);
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcGetCodeSize error: %s\n", hiprtcGetErrorString(result));
			free(code0);
			code0 = 0;
			deleteFFT(app);
			return FFT_ERROR_FAILED_TO_GET_CODE;
		}
		char* code = (char*)malloc(codeSize);
		if (!code) {
			free(code0);
			code0 = 0;
			deleteFFT(app);
			return FFT_ERROR_MALLOC_FAILED;
		}
		result = hiprtcGetCode(prog, code);
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcGetCode error: %s\n", hiprtcGetErrorString(result));
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteFFT(app);
			return FFT_ERROR_FAILED_TO_GET_CODE_SIZE;
		}
		//printf("%s\n", code);
		// Destroy the program.
		result = hiprtcDestroyProgram(&prog);
		if (result != HIPRTC_SUCCESS) {
			printf("hiprtcDestroyProgram error: %s\n", hiprtcGetErrorString(result));
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteFFT(app);
			return FFT_ERROR_FAILED_TO_DESTROY_PROGRAM;
		}
		hipError_t result2 = hipModuleLoadDataEx(&axis->FFTModule, code, 0, 0, 0);

		if (result2 != hipSuccess) {
			printf("hipModuleLoadDataEx error: %d\n", result2);
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteFFT(app);
			return FFT_ERROR_FAILED_TO_LOAD_MODULE;
		}
		result2 = hipModuleGetFunction(&axis->FFTKernel, axis->FFTModule, "FFT_main");
		if (result2 != hipSuccess) {
			printf("hipModuleGetFunction error: %d\n", result2);
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteFFT(app);
			return FFT_ERROR_FAILED_TO_GET_FUNCTION;
		}
		if (axis->layout.usedSharedMemory > app->configuration.sharedMemorySizeStatic) {
			result2 = hipFuncSetAttribute(axis->FFTKernel, hipFuncAttributeMaxDynamicSharedMemorySize, (int)axis->layout.usedSharedMemory);
			if (result2 != hipSuccess) {
				printf("hipFuncSetAttribute error: %d\n", result2);
				free(code);
				code = 0;
				free(code0);
				code0 = 0;
				deleteFFT(app);
				return FFT_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY;
			}
		}
		size_t size = (app->configuration.useUint64) ? sizeof(FFTLayoutUint64) : sizeof(FFTLayoutUint32);
		result2 = hipModuleGetGlobal(&axis->consts_addr, &size, axis->FFTModule, "consts");
		if (result2 != hipSuccess) {
			printf("hipModuleGetGlobal error: %d\n", result2);
			free(code);
			code = 0;
			free(code0);
			code0 = 0;
			deleteFFT(app);
			return FFT_ERROR_FAILED_TO_MODULE_GET_GLOBAL;
		}

		free(code0);
		code0 = 0;
	}
	if (axis->layout.axisSwapped) {//swap back for correct dispatch
		uint64_t temp = axis->axisBlock[1];
		axis->axisBlock[1] = axis->axisBlock[0];
		axis->axisBlock[0] = temp;
		axis->layout.axisSwapped = 0;
	}
	return resFFT;
}


FFTResult initializeFFT(FFTApplication* app, FFTConfiguration inputLaunchConfiguration) {
	if (inputLaunchConfiguration.doublePrecision != 0)	app->configuration.doublePrecision = inputLaunchConfiguration.doublePrecision;

	hipError_t res = hipSuccess;
	if (inputLaunchConfiguration.device == 0) {
		deleteFFT(app);
		return FFT_ERROR_INVALID_DEVICE;
	}
	app->configuration.device = inputLaunchConfiguration.device;

	int value = 0;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteFFT(app);
		return FFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxThreadsNum = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimX, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteFFT(app);
		return FFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupCount[0] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimY, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteFFT(app);
		return FFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupCount[1] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimZ, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteFFT(app);
		return FFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupCount[2] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimX, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteFFT(app);
		return FFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupSize[0] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimY, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteFFT(app);
		return FFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupSize[1] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimZ, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteFFT(app);
		return FFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.maxComputeWorkGroupSize[2] = value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlock, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteFFT(app);
		return FFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.sharedMemorySizeStatic = value;
	app->configuration.sharedMemorySize = (value > 65536) ? 65536 : value;
	res = hipDeviceGetAttribute(&value, hipDeviceAttributeWarpSize, app->configuration.device[0]);
	if (res != hipSuccess) {
		deleteFFT(app);
		return FFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
	}
	app->configuration.warpSize = value;
	app->configuration.sharedMemorySizePow2 = (uint64_t)pow(2, (uint64_t)log2(app->configuration.sharedMemorySize));
	
	app->configuration.coalescedMemory = 0 ? 64 : 32;
	app->configuration.useLUT = 1;
	app->configuration.regAd = 1;

	//set main parameters:
	if (inputLaunchConfiguration.FFTdim == 0) {
		deleteFFT(app);
		return FFT_ERROR_EMPTY_FFTdim;
	}
	app->configuration.FFTdim = inputLaunchConfiguration.FFTdim;
	if (inputLaunchConfiguration.size[0] == 0) {
		deleteFFT(app);
		return FFT_ERROR_EMPTY_size;
	}

	app->configuration.size[0] = inputLaunchConfiguration.size[0];

    if (inputLaunchConfiguration.bufferStride[0] == 0) app->configuration.bufferStride[0] = app->configuration.size[0];
	else app->configuration.bufferStride[0] = inputLaunchConfiguration.bufferStride[0];

    if (inputLaunchConfiguration.inputBufferStride[0] == 0) app->configuration.inputBufferStride[0] = app->configuration.size[0];
	else app->configuration.inputBufferStride[0] = inputLaunchConfiguration.inputBufferStride[0];

	if (inputLaunchConfiguration.outputBufferStride[0] == 0) app->configuration.outputBufferStride[0] = app->configuration.size[0];
	else app->configuration.outputBufferStride[0] = inputLaunchConfiguration.outputBufferStride[0];

	for (uint64_t i = 1; i < 3; i++) {
		if (inputLaunchConfiguration.size[i] == 0)
			app->configuration.size[i] = 1;
		else
			app->configuration.size[i] = inputLaunchConfiguration.size[i];

		if (inputLaunchConfiguration.bufferStride[i] == 0)
			app->configuration.bufferStride[i] = app->configuration.bufferStride[i - 1] * app->configuration.size[i];
		else
			app->configuration.bufferStride[i] = inputLaunchConfiguration.bufferStride[i];

		if (inputLaunchConfiguration.inputBufferStride[i] == 0)
			app->configuration.inputBufferStride[i] = app->configuration.inputBufferStride[i - 1] * app->configuration.size[i];
		else
			app->configuration.inputBufferStride[i] = inputLaunchConfiguration.inputBufferStride[i];

		if (inputLaunchConfiguration.outputBufferStride[i] == 0)
			app->configuration.outputBufferStride[i] = app->configuration.outputBufferStride[i - 1] * app->configuration.size[i];
		else
			app->configuration.outputBufferStride[i] = inputLaunchConfiguration.outputBufferStride[i];
	}



	if (inputLaunchConfiguration.bufferSize == 0) {
		deleteFFT(app);
		return FFT_ERROR_EMPTY_bufferSize;
	}
	app->configuration.bufferSize = inputLaunchConfiguration.bufferSize;
	app->configuration.buffer = inputLaunchConfiguration.buffer;

	if (inputLaunchConfiguration.userTempBuffer != 0)	app->configuration.userTempBuffer = inputLaunchConfiguration.userTempBuffer;

		// app->configuration.tempBufferNum = 1;
	app->configuration.tempBufferSize = (uint64_t*)malloc(sizeof(uint64_t));
	if (!app->configuration.tempBufferSize) {
		deleteFFT(app);
		return FFT_ERROR_MALLOC_FAILED;
	}
	app->configuration.tempBufferSize[0] = 0;

	for (uint64_t i = 0; i < 1; i++) {
		app->configuration.tempBufferSize[0] += app->configuration.bufferSize[i];
	}
	

	app->configuration.inputBufferSize = app->configuration.bufferSize;
	app->configuration.inputBuffer = app->configuration.buffer;
	
	app->configuration.outputBufferSize = app->configuration.bufferSize;
	app->configuration.outputBuffer = app->configuration.buffer;
	

	//set optional parameters:
	uint64_t checkBufferSizeFor64BitAddressing = 0;
	for (uint64_t i = 0; i < 1; i++) {
		checkBufferSizeFor64BitAddressing += app->configuration.bufferSize[i];
	}
	if (checkBufferSizeFor64BitAddressing >= (uint64_t)pow((uint64_t)2, (uint64_t)34)) app->configuration.useUint64 = 1;
	checkBufferSizeFor64BitAddressing = 0;
	for (uint64_t i = 0; i < 1; i++) {
		checkBufferSizeFor64BitAddressing += app->configuration.inputBufferSize[i];
	}
	if (checkBufferSizeFor64BitAddressing >= (uint64_t)pow((uint64_t)2, (uint64_t)34)) app->configuration.useUint64 = 1;

	checkBufferSizeFor64BitAddressing = 0;
	for (uint64_t i = 0; i < 1; i++) {
		checkBufferSizeFor64BitAddressing += app->configuration.outputBufferSize[i];
	}
	if (checkBufferSizeFor64BitAddressing >= (uint64_t)pow((uint64_t)2, (uint64_t)34)) app->configuration.useUint64 = 1;

	checkBufferSizeFor64BitAddressing = 0;
	for (uint64_t i = 0; i < 0; i++) {
		checkBufferSizeFor64BitAddressing += app->configuration.kernelSize[i];
	}
	if (checkBufferSizeFor64BitAddressing >= (uint64_t)pow((uint64_t)2, (uint64_t)34)) app->configuration.useUint64 = 1;
	if (inputLaunchConfiguration.useUint64 != 0)	app->configuration.useUint64 = inputLaunchConfiguration.useUint64;

	if (inputLaunchConfiguration.coalescedMemory != 0)	app->configuration.coalescedMemory = inputLaunchConfiguration.coalescedMemory;
	app->configuration.aimThreads = 128;
	if (inputLaunchConfiguration.aimThreads != 0)	app->configuration.aimThreads = inputLaunchConfiguration.aimThreads;
	app->configuration.numSharedBanks = 32;
	if (inputLaunchConfiguration.numSharedBanks != 0)	app->configuration.numSharedBanks = inputLaunchConfiguration.numSharedBanks;


	app->configuration.normalize = 0;
	if (inputLaunchConfiguration.normalize != 0)	app->configuration.normalize = inputLaunchConfiguration.normalize;
	if (inputLaunchConfiguration.makeForwardPlanOnly != 0)	app->configuration.makeForwardPlanOnly = inputLaunchConfiguration.makeForwardPlanOnly;
	if (inputLaunchConfiguration.makeInversePlanOnly != 0)	app->configuration.makeInversePlanOnly = inputLaunchConfiguration.makeInversePlanOnly;

	app->configuration.reorderFourStep = 1;
	app->configuration.regAd = 1;

	app->configuration.numberBatches = 1;
	if (inputLaunchConfiguration.numberBatches != 0)	app->configuration.numberBatches = inputLaunchConfiguration.numberBatches;


	app->configuration.maxCodeLength = 1000000;
	app->configuration.maxTempLength = 5000;

;

	FFTResult resFFT = FFT_SUCCESS;
	uint64_t initSharedMemory = app->configuration.sharedMemorySize;
	if (!app->configuration.makeForwardPlanOnly) {
		app->localFFTPlan_inverse = (FFTPlan*)calloc(1, sizeof(FFTPlan));
		if (app->localFFTPlan_inverse) {
			for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
				app->configuration.sharedMemorySize = ((app->configuration.size[i] & (app->configuration.size[i] - 1)) == 0) ? app->configuration.sharedMemorySizePow2 : initSharedMemory;
				resFFT = FFTScheduler(app, app->localFFTPlan_inverse, i, 0);
				if (resFFT != FFT_SUCCESS) {
					deleteFFT(app);
					return resFFT;
				}
				for (uint64_t j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++) {
					resFFT = FFTPlanAxis(app, app->localFFTPlan_inverse, i, j, 1);
					if (resFFT != FFT_SUCCESS) {
						deleteFFT(app);
						return resFFT;
					}
				}
			}
		}
		else {
			deleteFFT(app);
			return FFT_ERROR_MALLOC_FAILED;
		}
	}
	if (!app->configuration.makeInversePlanOnly) {
		app->localFFTPlan = (FFTPlan*)calloc(1, sizeof(FFTPlan));
		if (app->localFFTPlan) {
			for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
				app->configuration.sharedMemorySize = ((app->configuration.size[i] & (app->configuration.size[i] - 1)) == 0) ? app->configuration.sharedMemorySizePow2 : initSharedMemory;
				resFFT = FFTScheduler(app, app->localFFTPlan, i, 0);

				if (resFFT != FFT_SUCCESS) {
					deleteFFT(app);
					return resFFT;
				}
				for (uint64_t j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++) {
					resFFT = FFTPlanAxis(app, app->localFFTPlan, i, j, 0);
					
					if (resFFT != FFT_SUCCESS) {
						deleteFFT(app);
						return resFFT;
					}
				}
			}
		}
		else {
			deleteFFT(app);
			return FFT_ERROR_MALLOC_FAILED;
		}
	}
	return resFFT;
}

FFTResult dispatchEnhanced(FFTApplication* app, FFTAxis* axis, uint64_t* dispatchBlock, FFTLaunchArgs* launchArgs) {

	FFTResult resFFT = FFT_SUCCESS;

	launchArgs->args[0] = axis->inputBuffer;
	launchArgs->args[1] = axis->outputBuffer;  

	if (axis->layout.LUT) {
		launchArgs->args[2] = &axis->bufferLUT;
	}
	for(int i = 0; i < 3; i++){
		launchArgs->gridSize[i] = (unsigned int)dispatchBlock[i];
		launchArgs->blockSize[i] = (unsigned int)axis->layout.localSize[i];
	}
	launchArgs->sharedMem = (unsigned int)axis->layout.usedSharedMemory;
		
	//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",maxBlockSize[0], maxBlockSize[1], maxBlockSize[2], axis->layout.localSize[0], axis->layout.localSize[1], axis->layout.localSize[2]);
				
	return resFFT;
}

FFTResult FFTAppend(FFTApplication* app, int inverse, FFTLaunchParams* launchParams) {
	FFTResult resFFT = FFT_SUCCESS;


	uint64_t localSize0[3];
	if ((inverse != 1) && (app->configuration.makeInversePlanOnly)) return FFT_ERROR_ONLY_INVERSE_FFT_INITIALIZED;
	if ((inverse == 1) && (app->configuration.makeForwardPlanOnly)) return FFT_ERROR_ONLY_FORWARD_FFT_INITIALIZED;
	if ((inverse != 1) && (!app->configuration.makeInversePlanOnly) && (!app->localFFTPlan)) return FFT_ERROR_PLAN_NOT_INITIALIZED;
	if ((inverse == 1) && (!app->configuration.makeForwardPlanOnly) && (!app->localFFTPlan_inverse)) return FFT_ERROR_PLAN_NOT_INITIALIZED;

	if (inverse == 1) {
		localSize0[0] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0];
		localSize0[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[1][0];
		localSize0[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[2][0];
	}
	else {
		localSize0[0] = app->localFFTPlan->actualFFTSizePerAxis[0][0];
		localSize0[1] = app->localFFTPlan->actualFFTSizePerAxis[1][0];
		localSize0[2] = app->localFFTPlan->actualFFTSizePerAxis[2][0];
	}

	resFFT = FFTCheckBuffer(app, 0, 0, launchParams);
	if (resFFT != FFT_SUCCESS) {
		return resFFT;
	}


	if (inverse != 1) {
		for (int64_t l = (int64_t)app->localFFTPlan->numAxisUploads[0] - 1; l >= 0; l--) {
			FFTAxis* axis = &app->localFFTPlan->axes[0][l];
			FFTLaunchArgs* launchArgs = &app->localFFTPlan->launchArgs[0][l];
			resFFT = FFTUpdateBuffer(app, app->localFFTPlan, axis, 0, l, 0);
			if (resFFT != FFT_SUCCESS) return resFFT;
			
			uint64_t dispatchBlock[3];
			if (l == 0) {
				if (app->localFFTPlan->numAxisUploads[0] > 2) {
					dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->layout.dim / (double)axis->axisBlock[1]) / (double)app->localFFTPlan->axisSplit[0][1]) * app->localFFTPlan->axisSplit[0][1];
					dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
				}
				else {
					if (app->localFFTPlan->numAxisUploads[0] > 1) {
						dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->layout.dim / (double)axis->axisBlock[1]));
						dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
					}
					else {
						dispatchBlock[0] = app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->layout.dim;
						dispatchBlock[1] = (uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);

					}
				}
			}
			else {
				dispatchBlock[0] = (uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->layout.dim / (double)axis->axisBlock[0]);
				dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
			}
			dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[0][2];
			resFFT = dispatchEnhanced(app, axis, dispatchBlock, launchArgs);
		    if (resFFT != FFT_SUCCESS) return resFFT;	
		}
        
		if (app->configuration.FFTdim > 1) {
			for (int64_t l = (int64_t)app->localFFTPlan->numAxisUploads[1] - 1; l >= 0; l--) {
				FFTAxis* axis = &app->localFFTPlan->axes[1][l];
			    FFTLaunchArgs* launchArgs = &app->localFFTPlan->launchArgs[1][l];
			    resFFT = FFTUpdateBuffer(app, app->localFFTPlan, axis, 1, l, 0);
			    if (resFFT != FFT_SUCCESS) return resFFT;
			    uint64_t dispatchBlock[3];
				dispatchBlock[0] = (uint64_t)ceil(localSize0[1] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[1][1] / (double)axis->layout.dim);
				dispatchBlock[1] = 1;
				dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[1][2];
			    resFFT = dispatchEnhanced(app, axis, dispatchBlock, launchArgs);
				if (resFFT != FFT_SUCCESS) return resFFT;
			}	
		}
		//FFT axis 2
		if (app->configuration.FFTdim > 2) {
			for (int64_t l = (int64_t)app->localFFTPlan->numAxisUploads[2] - 1; l >= 0; l--) {
				FFTAxis* axis = &app->localFFTPlan->axes[2][l];
				FFTLaunchArgs* launchArgs = &app->localFFTPlan->launchArgs[2][l];
				resFFT = FFTUpdateBuffer(app, app->localFFTPlan, axis, 2, l, 0);
				if (resFFT != FFT_SUCCESS) return resFFT;
				uint64_t dispatchBlock[3];
				dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[2][2] / (double)axis->layout.dim);
				dispatchBlock[1] = 1;
				dispatchBlock[2] = app->localFFTPlan->actualFFTSizePerAxis[2][1];
				resFFT = dispatchEnhanced(app, axis, dispatchBlock, launchArgs);
				if (resFFT != FFT_SUCCESS) return resFFT;
			}
		}
	}

	if (inverse == 1) {

		if (app->configuration.FFTdim > 2) {
			for (int64_t l = (int64_t)app->localFFTPlan_inverse->numAxisUploads[2] - 1; l >= 0; l--) {
				if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[2] - 1 - l;
				FFTAxis* axis = &app->localFFTPlan_inverse->axes[2][l];
				FFTLaunchArgs* launchArgs = &app->localFFTPlan_inverse->launchArgs[2][l];
				resFFT = FFTUpdateBuffer(app, app->localFFTPlan_inverse, axis, 2, l, 1);
				if (resFFT != FFT_SUCCESS) return resFFT;
				uint64_t dispatchBlock[3];
				dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[2][2] / (double)axis->layout.dim);
				dispatchBlock[1] = 1;
				dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[2][1];
				resFFT = dispatchEnhanced(app, axis, dispatchBlock, launchArgs);
				if (resFFT != FFT_SUCCESS) return resFFT;
				if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[2] - 1 - l;
			}
			

		}
		if (app->configuration.FFTdim > 1) {
			for (int64_t l = (int64_t)app->localFFTPlan_inverse->numAxisUploads[1] - 1; l >= 0; l--) {
				if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[1] - 1 - l;
				FFTAxis* axis = &app->localFFTPlan_inverse->axes[1][l];
				FFTLaunchArgs* launchArgs = &app->localFFTPlan_inverse->launchArgs[1][l];
				resFFT = FFTUpdateBuffer(app, app->localFFTPlan_inverse, axis, 1, l, 1);
				if (resFFT != FFT_SUCCESS) return resFFT;
				uint64_t dispatchBlock[3];
				dispatchBlock[0] = (uint64_t)ceil(localSize0[1] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[1][1] / (double)axis->layout.dim);
				dispatchBlock[1] = 1;
				dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[1][2];
				resFFT = dispatchEnhanced(app, axis, dispatchBlock, launchArgs);
				if (resFFT != FFT_SUCCESS) return resFFT;
				if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[1] - 1 - l;
			}
		}
		//FFT axis 0
		for (int64_t l = (int64_t)app->localFFTPlan_inverse->numAxisUploads[0] - 1; l >= 0; l--) {
			if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[0] - 1 - l;
			FFTAxis* axis = &app->localFFTPlan_inverse->axes[0][l];
			FFTLaunchArgs* launchArgs = &app->localFFTPlan_inverse->launchArgs[0][l];
			resFFT = FFTUpdateBuffer(app, app->localFFTPlan_inverse, axis, 0, l, 1);
			if (resFFT != FFT_SUCCESS) return resFFT;
			uint64_t dispatchBlock[3];
			if (l == 0) {
				if (app->localFFTPlan_inverse->numAxisUploads[0] > 2) {
					dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->layout.dim / (double)axis->axisBlock[1]) / (double)app->localFFTPlan_inverse->axisSplit[0][1]) * app->localFFTPlan_inverse->axisSplit[0][1];
					dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
				}
				else {
					if (app->localFFTPlan_inverse->numAxisUploads[0] > 1) {
						dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->layout.dim / (double)axis->axisBlock[1]));
						dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
					}
					else {
						dispatchBlock[0] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->layout.dim;
						dispatchBlock[1] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
					}
				}
			}
			else {
				dispatchBlock[0] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->layout.dim / (double)axis->axisBlock[0]);
				dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
			}
			dispatchBlock[2] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][2];
			resFFT = dispatchEnhanced(app, axis, dispatchBlock, launchArgs);
			if (resFFT != FFT_SUCCESS) return resFFT;
			if (!app->configuration.reorderFourStep) l = app->localFFTPlan_inverse->numAxisUploads[0] - 1 - l;
		}
	}
	return resFFT;
}


FFTResult setFFTArgs(GPU* GPU, FFTApplication* app, FFTLaunchParams* launchParams, int inverse) {
	FFTResult resFFT = FFT_SUCCESS;
	resFFT = FFTAppend(app, inverse, launchParams);
	if (resFFT != FFT_SUCCESS) return resFFT;
	return resFFT;
}

hipError_t launchFFTKernel(FFTApplication* app,  int inverse){
	hipError_t res = hipSuccess;
	
	if(inverse != 1){
	for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
		for (int64_t j = (int64_t)app->localFFTPlan->numAxisUploads[i] - 1; j >= 0; j--) {
			FFTAxis* axis = &app->localFFTPlan->axes[i][j];
			FFTLaunchArgs *launchArgs = &app->localFFTPlan->launchArgs[i][j];
			launchArgs->args[0] = app->configuration.tempBuffer;
			launchArgs->args[1] = app->configuration.tempBuffer;
			if (j == (int64_t)app->localFFTPlan->numAxisUploads[i] - 1){
				launchArgs->args[0] = app->configuration.buffer;
			}
			if(j == 0){
			    launchArgs->args[1] = app->configuration.buffer;
			}
			res = hipModuleLaunchKernel(axis->FFTKernel,
						launchArgs->gridSize[0], launchArgs->gridSize[1], launchArgs->gridSize[2],     // grid dim
						launchArgs->blockSize[0], launchArgs->blockSize[1], launchArgs->blockSize[2],  // block dim
						launchArgs->sharedMem, 0,             // shared mem and stream
						launchArgs->args, 
						0);
		}		
	}
	}
	if(inverse == 1){
		for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
		for (int64_t j = (int64_t)app->localFFTPlan_inverse->numAxisUploads[i] - 1; j >= 0; j--) {
			FFTAxis* axis = &app->localFFTPlan_inverse->axes[i][j];
			FFTLaunchArgs *launchArgs = &app->localFFTPlan_inverse->launchArgs[i][j];
            launchArgs->args[0] = app->configuration.tempBuffer;
			launchArgs->args[1] = app->configuration.tempBuffer;
			if (j == (int64_t)app->localFFTPlan_inverse->numAxisUploads[i] - 1){
				launchArgs->args[0] = app->configuration.buffer;
			}
			if(j == 0){
			    launchArgs->args[1] = app->configuration.buffer;
			}
			res = hipModuleLaunchKernel(axis->FFTKernel,
						launchArgs->gridSize[0], launchArgs->gridSize[1], launchArgs->gridSize[2],     // grid dim
						launchArgs->blockSize[0], launchArgs->blockSize[1], launchArgs->blockSize[2],  // block dim
						launchArgs->sharedMem, 0,             // shared mem and stream
						launchArgs->args, 
						0);
		}		
		}
	}
	return res;
}











