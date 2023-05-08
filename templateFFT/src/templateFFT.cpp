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



FFTResult FFTCheckUpdateBufferSet(FFTApplication* app, FFTAxis* axis, uint64_t planStage, FFTLaunchParams* launchParams) {
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
		if (planStage) axis->specializationConstants.performBufferSetUpdate = 1;
		else {
			if (!app->configuration.makeInversePlanOnly) {
				for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
					for (uint64_t j = 0; j < app->localFFTPlan->numAxisUploads[i]; j++)
						app->localFFTPlan->axes[i][j].specializationConstants.performBufferSetUpdate = 1;
				}
			}
			if (!app->configuration.makeForwardPlanOnly) {
				for (uint64_t i = 0; i < app->configuration.FFTdim; i++) {
					for (uint64_t j = 0; j < app->localFFTPlan_inverse->numAxisUploads[i]; j++)
						app->localFFTPlan_inverse->axes[i][j].specializationConstants.performBufferSetUpdate = 1;
				}
			}
		}
	}
	return FFT_SUCCESS;
}





FFTResult FFTUpdateBufferSet(FFTApplication* app, FFTPlan* FFTPlan, FFTAxis* axis, uint64_t axis_id, uint64_t axis_upload_id, uint64_t inverse) {
	if (axis->specializationConstants.performBufferSetUpdate) {
		uint64_t storageComplexSize;
		if (app->configuration.doublePrecision) storageComplexSize = (2 * sizeof(double));

		for (uint64_t i = 0; i < axis->numBindings; ++i) {
			for (uint64_t j = 0; j < axis->specializationConstants.numBuffersBound[i]; ++j) {
				if (i == 0) {
					uint64_t bufferId = 0;
					uint64_t offset = j;
					if ((FFTPlan->axes[axis_id]->specializationConstants.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
						if (axis_upload_id > 0) axis->inputBuffer = app->configuration.buffer;
						else axis->inputBuffer = app->configuration.tempBuffer;
					else axis->inputBuffer = app->configuration.buffer;
				}
				if (i == 1) {
					if ((FFTPlan->axes[axis_id]->specializationConstants.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1)) {
						if (axis_upload_id == 1) axis->outputBuffer = app->configuration.tempBuffer;
						else axis->outputBuffer = app->configuration.buffer;
					}
					else axis->outputBuffer = app->configuration.buffer;
				}
			}
		}
		axis->specializationConstants.performBufferSetUpdate = 0;
	}
	return FFT_SUCCESS;
}

FFTResult AppendLine(FFTSpecializationConstantsLayout* sc) {
	if (sc->tempLen < 0) return FFT_ERROR_INSUFFICIENT_TEMP_BUFFER;
	if (sc->currentLen + sc->tempLen > sc->maxCodeLength) return FFT_ERROR_INSUFFICIENT_CODE_BUFFER;
	sc->currentLen += sprintf(sc->output + sc->currentLen, "%s", sc->tempStr);
	return FFT_SUCCESS;
};

FFTResult MulComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2, const char* temp) {
	FFTResult res = FFT_SUCCESS;
	if (strcmp(out, in_1) && strcmp(out, in_2)) {
		sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = %s.x * %s.x - %s.y * %s.y;\n\
	%s.y = %s.y * %s.x + %s.x * %s.y;\n", out, in_1, in_2, in_1, in_2, out, in_1, in_2, in_1, in_2);
	}
	else {
		if (temp) {
			sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = %s.x * %s.x - %s.y * %s.y;\n\
	%s.y = %s.y * %s.x + %s.x * %s.y;\n\
	%s = %s;\n", temp, in_1, in_2, in_1, in_2, temp, in_1, in_2, in_1, in_2, out, temp);
		}
		else
			return FFT_ERROR_NULL_TEMP_PASSED;
	}
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult SubComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = %s.x - %s.x;\n\
	%s.y = %s.y - %s.y;\n", out, in_1, in_2, out, in_1, in_2);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult AddComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = %s.x + %s.x;\n\
	%s.y = %s.y + %s.y;\n", out, in_1, in_2, out, in_1, in_2);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};
FFTResult AddComplexInv(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = - %s.x - %s.x;\n\
	%s.y = - %s.y - %s.y;\n", out, in_1, in_2, out, in_1, in_2);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult FMAComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = fma(%s.x, %s, %s.x);\n\
	%s.y = fma(%s.y, %s, %s.y);\n", out, in_1, in_num, in_2, out, in_1, in_num, in_2);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult MulComplexNumber(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = %s.x * %s;\n\
	%s.y = %s.y * %s;\n", out, in_1, in_num, out, in_1, in_num);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};
FFTResult MovComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s = %s;\n", out, in);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};
FFTResult ShuffleComplex(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2, const char* temp) {
	FFTResult res = FFT_SUCCESS;
	if (strcmp(out, in_2)) {
		sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = %s.x - %s.y;\n\
	%s.y = %s.y + %s.x;\n", out, in_1, in_2, out, in_1, in_2);
	}
	else {
		if (temp) {
			sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = %s.x - %s.y;\n\
	%s.y = %s.x + %s.y;\n\
	%s = %s;\n", temp, in_1, in_2, temp, in_1, in_2, out, temp);
		}
		else
			return FFT_ERROR_NULL_TEMP_PASSED;
	}
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult ShuffleComplexInv(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2, const char* temp) {
	FFTResult res = FFT_SUCCESS;
	if (strcmp(out, in_2)) {
		sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = %s.x + %s.y;\n\
	%s.y = %s.y - %s.x;\n", out, in_1, in_2, out, in_1, in_2);
	}
	else {
		if (temp) {
			sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = %s.x + %s.y;\n\
	%s.y = %s.x - %s.y;\n\
	%s = %s;\n", temp, in_1, in_2, temp, in_1, in_2, out, temp);
		}
		else
			return FFT_ERROR_NULL_TEMP_PASSED;
	}
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult DivComplexNumber(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s.x = %s.x / %s;\n\
	%s.y = %s.y / %s;\n", out, in_1, in_num, out, in_1, in_num);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult AddReal(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s = %s + %s;\n", out, in_1, in_2);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult MovReal(FFTSpecializationConstantsLayout* sc, const char* out, const char* in) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s = %s;\n", out, in);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult ModReal(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_num) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s = %s %% %s;\n", out, in_1, in_num);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult SubReal(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s = %s - %s;\n", out, in_1, in_2);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult MulReal(FFTSpecializationConstantsLayout* sc, const char* out, const char* in_1, const char* in_2) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s = %s * %s;\n", out, in_1, in_2);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult SharedStore(FFTSpecializationConstantsLayout* sc, const char* id, const char* in) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	sdata[%s] = %s;\n", id, in);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};

FFTResult SharedLoad(FFTSpecializationConstantsLayout* sc, const char* out, const char* id) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "\
	%s = sdata[%s];\n", out, id);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
};


FFTResult inlineRadixKernelFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t radix, uint64_t stageSize, double stageAngle, char** regID) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char LFending[4] = "";
	if (!strcmp(floatType, "float")) sprintf(LFending, "f");

	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	char cosDef[20] = "__cosf";
	char sinDef[20] = "__sinf";
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");

	char* temp = sc->temp;
	char* w = sc->w;
	char* iw = sc->iw;
	char convolutionInverse[30] = "";
	if (sc->convolutionStep) sprintf(convolutionInverse, ", %s inverse", uintType);
	switch (radix) {
	case 2: {

		if (sc->LUT) {
			sc->tempLen = sprintf(sc->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if (!sc->inverse) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "float")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(angle);\n", w, cosDef);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(angle);\n", w, sinDef);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
			if (!strcmp(floatType, "double")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s = sincos_20(angle);\n", w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(sc, temp, regID[1], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[1], regID[0], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[0], regID[0], temp);
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

		if (sc->LUT) {
			sc->tempLen = sprintf(sc->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if (!sc->inverse) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "float")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(angle*%.17f%s);\n", w, cosDef, 4.0 / 3.0, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(angle*%.17f%s);\n", w, sinDef, 4.0 / 3.0, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				//sc->tempLen = sprintf(sc->tempStr, "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 4.0 / 3.0, 4.0 / 3.0);
			}
			if (!strcmp(floatType, "double")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s = sincos_20(angle*%.17f%s);\n", w, 4.0 / 3.0, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(sc, sc->locID[2], regID[2], w, 0);
		/*sc->tempLen = sprintf(sc->tempStr, "\
loc_2.x = temp%s.x * w.x - temp%s.y * w.y;\n\
loc_2.y = temp%s.y * w.x + temp%s.x * w.y;\n", regID[2], regID[2], regID[2], regID[2]);*/
		if (sc->LUT) {
			sc->tempLen = sprintf(sc->tempStr, "	%s = twiddleLUT[LUTId+%" PRIu64 "];\n", w, stageSize);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if (!sc->inverse) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "float")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(angle*%.17f%s);\n", w, cosDef, 2.0 / 3.0, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(angle*%.17f%s);\n", w, sinDef, 2.0 / 3.0, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				//sc->tempLen = sprintf(sc->tempStr, "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 / 3.0, 2.0 / 3.0);
			}
			if (!strcmp(floatType, "double")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s=sincos_20(angle*%.17f%s);\n", w, 2.0 / 3.0, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(sc, sc->locID[1], regID[1], w, 0);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(sc, regID[1], sc->locID[1], sc->locID[2]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[2], sc->locID[1], sc->locID[2]);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(sc, sc->locID[0], regID[0], regID[1]);
		if (res != FFT_SUCCESS) return res;
		res = FMAComplex(sc, sc->locID[1], regID[1], tf[0], regID[0]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, sc->locID[2], regID[2], tf[1]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, regID[0], sc->locID[0]);
		if (res != FFT_SUCCESS) return res;


		if (stageAngle < 0)
		{
			res = ShuffleComplex(sc, regID[1], sc->locID[1], sc->locID[2], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplexInv(sc, regID[2], sc->locID[1], sc->locID[2], 0);
			if (res != FFT_SUCCESS) return res;

		}
		else {
			res = ShuffleComplexInv(sc, regID[1], sc->locID[1], sc->locID[2], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplex(sc, regID[2], sc->locID[1], sc->locID[2], 0);
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
		if (sc->LUT) {
			sc->tempLen = sprintf(sc->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if (!sc->inverse) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "float")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(angle);\n", w, cosDef);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(angle);\n", w, sinDef);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
			if (!strcmp(floatType, "double")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s = sincos_20(angle);\n", w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(sc, temp, regID[2], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[2], regID[0], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[0], regID[0], temp);
		if (res != FFT_SUCCESS) return res;
		res = MulComplex(sc, temp, regID[3], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[3], regID[1], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[1], regID[1], temp);
		if (res != FFT_SUCCESS) return res;

		if (sc->LUT) {
			sc->tempLen = sprintf(sc->tempStr, "	%s=twiddleLUT[LUTId+%" PRIu64 "];\n", w, stageSize);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if (!sc->inverse) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "float")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(0.5%s*angle);\n", w, cosDef, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(0.5%s*angle);\n", w, sinDef, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
			if (!strcmp(floatType, "double")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s=normalize(%s + %s(1.0, 0.0));\n", w, w, vecType);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(sc, temp, regID[1], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[1], regID[0], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[0], regID[0], temp);
		if (res != FFT_SUCCESS) return res;

		if (stageAngle < 0) {
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s.x;", temp, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s.y;\n", w, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.x;\n", w, temp);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s.x;", temp, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = -%s.y;\n", w, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s.x;\n", w, temp);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			//sc->tempLen = sprintf(sc->tempStr, "	w = %s(-w.y, w.x);\n\n", vecType);
		}
		res = MulComplex(sc, temp, regID[3], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[3], regID[2], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[2], regID[2], temp);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, temp, regID[1]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, regID[1], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, regID[2], temp);
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
				if (sc->LUT) {
					sc->tempLen = sprintf(sc->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (!sc->inverse) {
						sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					if (!strcmp(floatType, "float")) {
						sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(angle*%.17f%s);\n", w, cosDef, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(angle*%.17f%s);\n", w, sinDef, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						
					}
					if (!strcmp(floatType, "double")) {
						sc->tempLen = sprintf(sc->tempStr, "	%s = sincos_20(angle*%.17f%s);\n", w, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			else {
				if (sc->LUT) {
					sc->tempLen = sprintf(sc->tempStr, "	%s = twiddleLUT[LUTId+%" PRIu64 "];\n", w, (radix - 1 - i) * stageSize);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (!sc->inverse) {
						sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					if (!strcmp(floatType, "float")) {
						sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(angle*%.17f%s);\n", w, cosDef, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(angle*%.17f%s);\n", w, sinDef, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					
					}
					if (!strcmp(floatType, "double")) {
						sc->tempLen = sprintf(sc->tempStr, "	%s = sincos_20(angle*%.17f%s);\n", w, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			res = MulComplex(sc, sc->locID[i], regID[i], w, 0);
			if (res != FFT_SUCCESS) return res;

		}
		res = AddComplex(sc, regID[1], sc->locID[1], sc->locID[4]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[2], sc->locID[2], sc->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[3], sc->locID[2], sc->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[4], sc->locID[1], sc->locID[4]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, sc->locID[3], regID[1], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[4], regID[3], regID[4]);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(sc, sc->locID[0], regID[0], regID[1]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[0], sc->locID[0], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = FMAComplex(sc, sc->locID[1], regID[1], tf[0], regID[0]);
		if (res != FFT_SUCCESS) return res;
		res = FMAComplex(sc, sc->locID[2], regID[2], tf[0], regID[0]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, regID[3], regID[3], tf[1]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, regID[4], regID[4], tf[2]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, sc->locID[3], sc->locID[3], tf[3]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, sc->locID[4], sc->locID[4], tf[4]);
		if (res != FFT_SUCCESS) return res;

		res = SubComplex(sc, sc->locID[1], sc->locID[1], sc->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[2], sc->locID[2], sc->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[3], regID[3], sc->locID[4]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[4], sc->locID[4], regID[4]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, regID[0], sc->locID[0]);
		if (res != FFT_SUCCESS) return res;


		if (stageAngle < 0)
		{
			res = ShuffleComplex(sc, regID[1], sc->locID[1], sc->locID[4], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplex(sc, regID[2], sc->locID[2], sc->locID[3], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplexInv(sc, regID[3], sc->locID[2], sc->locID[3], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplexInv(sc, regID[4], sc->locID[1], sc->locID[4], 0);
			if (res != FFT_SUCCESS) return res;

		}
		else {
			res = ShuffleComplexInv(sc, regID[1], sc->locID[1], sc->locID[4], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplexInv(sc, regID[2], sc->locID[2], sc->locID[3], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplex(sc, regID[3], sc->locID[2], sc->locID[3], 0);
			if (res != FFT_SUCCESS) return res;
			res = ShuffleComplex(sc, regID[4], sc->locID[1], sc->locID[4], 0);
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
				if (sc->LUT) {
					sc->tempLen = sprintf(sc->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (!sc->inverse) {
						sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					if (!strcmp(floatType, "float")) {
						sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(angle*%.17f%s);\n", w, cosDef, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(angle*%.17f%s);\n", w, sinDef, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						//sc->tempLen = sprintf(sc->tempStr, "	w = %s(cos(angle*%.17f), sin(angle*%.17f));\n\n", vecType, 2.0 * i / radix, 2.0 * i / radix);
					}
					if (!strcmp(floatType, "double")) {
						sc->tempLen = sprintf(sc->tempStr, "	%s = sincos_20(angle*%.17f%s);\n", w, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			else {
				if (sc->LUT) {
					sc->tempLen = sprintf(sc->tempStr, "	%s = twiddleLUT[LUTId+%" PRIu64 "];\n\n", w, (radix - 1 - i) * stageSize);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (!sc->inverse) {
						sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					if (!strcmp(floatType, "float")) {
						sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(angle*%.17f%s);\n", w, cosDef, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(angle*%.17f%s);\n", w, sinDef, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						
					}
					if (!strcmp(floatType, "double")) {
						sc->tempLen = sprintf(sc->tempStr, "	%s = sincos_20(angle*%.17f%s);\n", w, 2.0 * i / radix, LFending);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			res = MulComplex(sc, sc->locID[i], regID[i], w, 0);
			if (res != FFT_SUCCESS) return res;

		}
		res = MovComplex(sc, sc->locID[0], regID[0]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[0], sc->locID[1], sc->locID[6]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[1], sc->locID[1], sc->locID[6]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[2], sc->locID[2], sc->locID[5]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[3], sc->locID[2], sc->locID[5]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[4], sc->locID[4], sc->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[5], sc->locID[4], sc->locID[3]);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(sc, sc->locID[5], regID[1], regID[3]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[5], sc->locID[5], regID[5]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[1], regID[0], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[1], sc->locID[1], regID[4]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[0], sc->locID[0], sc->locID[1]);
		if (res != FFT_SUCCESS) return res;

		res = SubComplex(sc, sc->locID[2], regID[0], regID[4]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, sc->locID[3], regID[4], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, sc->locID[4], regID[2], regID[0]);
		if (res != FFT_SUCCESS) return res;

		res = SubComplex(sc, regID[0], regID[1], regID[5]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[2], regID[5], regID[3]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[4], regID[3], regID[1]);
		if (res != FFT_SUCCESS) return res;


		res = MulComplexNumber(sc, sc->locID[1], sc->locID[1], tf[0]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, sc->locID[2], sc->locID[2], tf[1]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, sc->locID[3], sc->locID[3], tf[2]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, sc->locID[4], sc->locID[4], tf[3]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, sc->locID[5], sc->locID[5], tf[4]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, regID[0], regID[0], tf[5]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, regID[2], regID[2], tf[6]);
		if (res != FFT_SUCCESS) return res;
		res = MulComplexNumber(sc, regID[4], regID[4], tf[7]);
		if (res != FFT_SUCCESS) return res;


		res = SubComplex(sc, regID[5], regID[4], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplexInv(sc, regID[6], regID[4], regID[0]);
		if (res != FFT_SUCCESS) return res;
		res =  AddComplex(sc, regID[4], regID[0], regID[2]);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(sc, regID[0], sc->locID[0], sc->locID[1]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[1], sc->locID[2], sc->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[2], sc->locID[4], sc->locID[3]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplexInv(sc, regID[3], sc->locID[2], sc->locID[4]);
		if (res != FFT_SUCCESS) return res;

		res = AddComplex(sc, sc->locID[1], regID[0], regID[1]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[2], regID[0], regID[2]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[3], regID[0], regID[3]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[4], regID[4], sc->locID[5]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[6], regID[6], sc->locID[5]);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, sc->locID[5], sc->locID[5], regID[5]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, regID[0], sc->locID[0]);
		if (res != FFT_SUCCESS) return res;

		res = ShuffleComplexInv(sc, regID[1], sc->locID[1], sc->locID[4], 0);
		if (res != FFT_SUCCESS) return res;
		res = ShuffleComplexInv(sc, regID[2], sc->locID[3], sc->locID[6], 0);
		if (res != FFT_SUCCESS) return res;
		res = ShuffleComplex(sc, regID[3], sc->locID[2], sc->locID[5], 0);
		if (res != FFT_SUCCESS) return res;
		res = ShuffleComplexInv(sc, regID[4], sc->locID[2], sc->locID[5], 0);
		if (res != FFT_SUCCESS) return res;
		res = ShuffleComplex(sc, regID[5], sc->locID[3], sc->locID[6], 0);
		if (res != FFT_SUCCESS) return res;
		res = ShuffleComplex(sc, regID[6], sc->locID[1], sc->locID[4], 0);
		if (res != FFT_SUCCESS) return res;


		for (uint64_t i = 0; i < 8; i++) {
			free(tf[i]);
			tf[i] = 0;
		}
		break;
	}
	case 8: {
		
		if (sc->LUT) {
			sc->tempLen = sprintf(sc->tempStr, "	%s = twiddleLUT[LUTId];\n", w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if (!sc->inverse) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "float")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(angle);\n", w, cosDef);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(angle);\n", w, sinDef);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
			if (!strcmp(floatType, "double")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s = sincos_20(angle);\n", w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		for (uint64_t i = 0; i < 4; i++) {
			res = MulComplex(sc, temp, regID[i + 4], w, 0);
			if (res != FFT_SUCCESS) return res;
			res = SubComplex(sc, regID[i + 4], regID[i], temp);
			if (res != FFT_SUCCESS) return res;
			res = AddComplex(sc, regID[i], regID[i], temp);
			if (res != FFT_SUCCESS) return res;

		}
		if (sc->LUT) {
			sc->tempLen = sprintf(sc->tempStr, "	%s=twiddleLUT[LUTId+%" PRIu64 "];\n\n", w, stageSize);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if (!sc->inverse) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "float")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(0.5%s*angle);\n", w, cosDef, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(0.5%s*angle);\n", w, sinDef, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
			if (!strcmp(floatType, "double")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s=normalize(%s + %s(1.0, 0.0));\n", w, w, vecType);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		for (uint64_t i = 0; i < 2; i++) {
			res = MulComplex(sc, temp, regID[i + 2], w, 0);
			if (res != FFT_SUCCESS) return res;
			res = SubComplex(sc, regID[i + 2], regID[i], temp);
			if (res != FFT_SUCCESS) return res;
			res = AddComplex(sc, regID[i], regID[i], temp);
			if (res != FFT_SUCCESS) return res;

		}
		if (stageAngle < 0) {
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s.y;\n", iw, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.x;\n", iw, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			//sc->tempLen = sprintf(sc->tempStr, "	w = %s(w.y, -w.x);\n\n", vecType);
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = -%s.y;\n", iw, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s.x;\n", iw, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			//sc->tempLen = sprintf(sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}

		for (uint64_t i = 4; i < 6; i++) {
			res = MulComplex(sc, temp, regID[i + 2], iw, 0);
			if (res != FFT_SUCCESS) return res;
			res = SubComplex(sc, regID[i + 2], regID[i], temp);
			if (res != FFT_SUCCESS) return res;
			res = AddComplex(sc, regID[i], regID[i], temp);
			if (res != FFT_SUCCESS) return res;

		}

		if (sc->LUT) {
			sc->tempLen = sprintf(sc->tempStr, "	%s=twiddleLUT[LUTId+%" PRIu64 "];\n\n", w, 2 * stageSize);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if (!sc->inverse) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.y;\n", w, w);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			if (!strcmp(floatType, "float")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s(0.25%s*angle);\n", w, cosDef, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s(0.25%s*angle);\n", w, sinDef, LFending);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				
			}
			if (!strcmp(floatType, "double")) {
				sc->tempLen = sprintf(sc->tempStr, "	%s=normalize(%s + %s(1.0, 0.0));\n", w, w, vecType);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		res = MulComplex(sc, temp, regID[1], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[1], regID[0], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[0], regID[0], temp);
		if (res != FFT_SUCCESS) return res;

		if (stageAngle < 0) {
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s.y;\n", iw, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.x;\n", iw, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = -%s.y;\n", iw, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s.x;\n", iw, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			//sc->tempLen = sprintf(sc->tempStr, "	iw = %s(-w.y, w.x);\n\n", vecType);
		}
		res = MulComplex(sc, temp, regID[3], iw, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[3], regID[2], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[2], regID[2], temp);
		if (res != FFT_SUCCESS) return res;
		/*sc->tempLen = sprintf(sc->tempStr, "\
temp.x = temp%s.x * iw.x - temp%s.y * iw.y;\n\
temp.y = temp%s.y * iw.x + temp%s.x * iw.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", regID[3], regID[3], regID[3], regID[3], regID[3], regID[2], regID[2], regID[2]);*/
		if (stageAngle < 0) {
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s.x * loc_SQRT1_2 + %s.y * loc_SQRT1_2;\n", iw, w, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s.y * loc_SQRT1_2 - %s.x * loc_SQRT1_2;\n\n", iw, w, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s.x * loc_SQRT1_2 - %s.y * loc_SQRT1_2;\n", iw, w, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s.y * loc_SQRT1_2 + %s.x * loc_SQRT1_2;\n\n", iw, w, w);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}
		res = MulComplex(sc, temp, regID[5], iw, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[5], regID[4], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[4], regID[4], temp);
		if (res != FFT_SUCCESS) return res;
		/*sc->tempLen = sprintf(sc->tempStr, "\
temp.x = temp%s.x * iw.x - temp%s.y * iw.y;\n\
temp.y = temp%s.y * iw.x + temp%s.x * iw.y;\n\
temp%s = temp%s - temp;\n\
temp%s = temp%s + temp;\n\n", regID[5], regID[5], regID[5], regID[5], regID[5], regID[4], regID[4], regID[4]);*/
		if (stageAngle < 0) {
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = %s.y;\n", w, iw);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.y = -%s.x;\n", w, iw);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			//sc->tempLen = sprintf(sc->tempStr, "	w = %s(iw.y, -iw.x);\n\n", vecType);
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "	%s.x = -%s.y;\n", w, iw);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	%s.y = %s.x;\n", w, iw);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			//sc->tempLen = sprintf(sc->tempStr, "	w = %s(-iw.y, iw.x);\n\n", vecType);
		}
		res = MulComplex(sc, temp, regID[7], w, 0);
		if (res != FFT_SUCCESS) return res;
		res = SubComplex(sc, regID[7], regID[6], temp);
		if (res != FFT_SUCCESS) return res;
		res = AddComplex(sc, regID[6], regID[6], temp);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, temp, regID[1]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, regID[1], regID[4]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, regID[4], temp);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, temp, regID[3]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, regID[3], regID[6]);
		if (res != FFT_SUCCESS) return res;
		res = MovComplex(sc, regID[6], temp);
		if (res != FFT_SUCCESS) return res;

		break;
	}


	}
	return res;
};


FFTResult appendExtensions(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeInputMemory, const char* floatTypeOutputMemory, const char* floatTypeKernelMemory) {
	FFTResult res = FFT_SUCCESS;

	sc->tempLen = sprintf(sc->tempStr, "\
#include <hip/hip_runtime.h>\n");
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;

	return res;
}
FFTResult appendPushConstant(FFTSpecializationConstantsLayout* sc, const char* type, const char* name) {
	FFTResult res = FFT_SUCCESS;
	sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", type, name);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
}

FFTResult appendConstant(FFTSpecializationConstantsLayout* sc, const char* type, const char* name, const char* defaultVal, const char* LFending) {
	FFTResult res = FFT_SUCCESS;

	sc->tempLen = sprintf(sc->tempStr, "const %s %s = %s%s;\n", type, name, defaultVal, LFending);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
}

FFTResult AppendLineFromInput(FFTSpecializationConstantsLayout* sc, const char* in) {
	//appends code line stored in tempStr to generated code
	if (sc->currentLen + (int64_t)strlen(in) > sc->maxCodeLength) return FFT_ERROR_INSUFFICIENT_CODE_BUFFER;
	sc->currentLen += sprintf(sc->output + sc->currentLen, "%s", in);
	return FFT_SUCCESS;
};

FFTResult appendConstantsFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType) {
	FFTResult res = FFT_SUCCESS;
	char LFending[4] = "";
	if (!strcmp(floatType, "float")) sprintf(LFending, "f");
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");

	res = appendConstant(sc, floatType, "loc_PI", "3.1415926535897932384626433832795", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "loc_SQRT1_2", "0.70710678118654752440084436210485", LFending);
	if (res != FFT_SUCCESS) return res;
	return res;
}




FFTResult appendSinCos20(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType) {
	FFTResult res = FFT_SUCCESS;
	char functionDefinitions[100] = "";
	char vecType[30];
	char LFending[4] = "";
	if (!strcmp(floatType, "float")) sprintf(LFending, "f");


	if (!strcmp(floatType, "half")) sprintf(vecType, "f16vec2");
	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");
	sprintf(functionDefinitions, "__device__ static __inline__ ");

	res = appendConstant(sc, floatType, "loc_2_PI", "0.63661977236758134307553505349006", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "loc_PI_2", "1.5707963267948966192313216916398", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "a1", "0.99999999999999999999962122687403772", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "a3", "-0.166666666666666666637194166219637268", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "a5", "0.00833333333333333295212653322266277182", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "a7", "-0.000198412698412696489459896530659927773", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "a9", "2.75573192239364018847578909205399262e-6", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "a11", "-2.50521083781017605729370231280411712e-8", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "a13", "1.60590431721336942356660057796782021e-10", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "a15", "-7.64712637907716970380859898835680587e-13", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "a17", "2.81018528153898622636194976499656274e-15", LFending);
	if (res != FFT_SUCCESS) return res;
	res = appendConstant(sc, floatType, "ab", "-7.97989713648499642889739108679114937e-18", LFending);
	if (res != FFT_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "\
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
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
}

FFTResult appendConversion(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeDifferent) {
	FFTResult res = FFT_SUCCESS;

	char functionDefinitions[100] = "";
	char vecType[30];
	char vecTypeDifferent[30];
	sprintf(functionDefinitions, "__device__ static __inline__ ");

	if (!strcmp(floatType, "half")) sprintf(vecType, "f16vec2");
	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	if (!strcmp(floatTypeDifferent, "half")) sprintf(vecTypeDifferent, "f16vec2");
	if (!strcmp(floatTypeDifferent, "float")) sprintf(vecTypeDifferent, "float2");
	if (!strcmp(floatTypeDifferent, "double")) sprintf(vecTypeDifferent, "double2");
	sc->tempLen = sprintf(sc->tempStr, "\
%s%s conv_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x = (%s) input.x;\n\
	ret_val.y = (%s) input.y;\n\
	return ret_val;\n\
}\n\n", functionDefinitions, vecType, vecType, vecTypeDifferent, vecType, floatType, floatType);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "\
%s%s conv_%s(%s input)\n\
{\n\
	%s ret_val;\n\
	ret_val.x = (%s) input.x;\n\
	ret_val.y = (%s) input.y;\n\
	return ret_val;\n\
}\n\n", functionDefinitions, vecTypeDifferent, vecTypeDifferent, vecType, vecTypeDifferent, floatTypeDifferent, floatTypeDifferent);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
}



FFTResult appendPushConstantsFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType) {
	FFTResult res = FFT_SUCCESS;

	sc->tempLen = sprintf(sc->tempStr, "	typedef struct {\n");
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	res = appendPushConstant(sc, uintType, "coordinate");
	if (res != FFT_SUCCESS) return res;
	res = appendPushConstant(sc, uintType, "batchID");
	if (res != FFT_SUCCESS) return res;
	res = appendPushConstant(sc, uintType, "workGroupShiftX");
	if (res != FFT_SUCCESS) return res;
	res = appendPushConstant(sc, uintType, "workGroupShiftY");
	if (res != FFT_SUCCESS) return res;
	res = appendPushConstant(sc, uintType, "workGroupShiftZ");
	if (res != FFT_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	}PushConsts;\n");
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	__constant__ PushConsts consts;\n");
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;

	return res;
}


FFTResult appendInputLayoutFFT(FFTSpecializationConstantsLayout* sc, uint64_t id, const char* floatTypeMemory, uint64_t inputType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	switch (inputType) {
	case 0: case 1: case 2: case 3: case 4: case 6: {
		if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double2");
		break;
	}
	case 5: case 120: case 121: case 130: case 131: case 140: case 141: case 142: case 143:
	{
		if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "float16_t");
		if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float");
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double");
		break;
	}
	}
	return res;
}

FFTResult appendOutputLayoutFFT(FFTSpecializationConstantsLayout* sc, uint64_t id, const char* floatTypeMemory, uint64_t outputType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	switch (outputType) {
	case 0: case 1: case 2: case 3: case 4: case 5: {


		if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "f16vec2");
		if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float2");
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double2");
		break;
	}
	case 6: case 120: case 121: case 130: case 131: case 140: case 141: case 142: case 143:
	{
		if (!strcmp(floatTypeMemory, "half")) sprintf(vecType, "float16_t");
		if (!strcmp(floatTypeMemory, "float")) sprintf(vecType, "float");
		if (!strcmp(floatTypeMemory, "double")) sprintf(vecType, "double");
		break;
	}
	}
	return res;
}

FFTResult appendLUTLayoutFFT(FFTSpecializationConstantsLayout* sc, uint64_t id, const char* floatType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	return res;
}

FFTResult appendSharedMemoryFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t sharedType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char sharedDefinitions[20] = "";
	uint64_t vecSize = 1;
	uint64_t maxSequenceSharedMemory = 0;
	if (!strcmp(floatType, "float"))
	{
		sprintf(vecType, "float2");
		sprintf(sharedDefinitions, "__shared__");
		vecSize = 8;
	}
	if (!strcmp(floatType, "double")) {
		sprintf(vecType, "double2");
		sprintf(sharedDefinitions, "__shared__");
		vecSize = 16;
	}
	maxSequenceSharedMemory = sc->sharedMemSize / vecSize;
	//maxSequenceSharedMemoryPow2 = sc->sharedMemSizePow2 / vecSize;
	uint64_t mergeR2C = (sc->mergeSequencesR2C && (sc->axis_id == 0)) ? 2 : 0;
	switch (sharedType) {
	case 0: case 5: case 6: case 120: case 130: case 140: case 142:
	{
		sc->resolveBankConflictFirstStages = 0;
		sc->sharedStrideBankConflictFirstStages = ((sc->fftDim > sc->numSharedBanks / 2) && ((sc->fftDim & (sc->fftDim - 1)) == 0)) ? sc->fftDim / sc->registerBoost * (sc->numSharedBanks / 2 + 1) / (sc->numSharedBanks / 2) : sc->fftDim / sc->registerBoost;
		sc->sharedStrideReadWriteConflict = ((sc->numSharedBanks / 2 <= sc->localSize[1])) ? sc->fftDim / sc->registerBoost + 1 : sc->fftDim / sc->registerBoost + (sc->numSharedBanks / 2) / sc->localSize[1];
		if (sc->sharedStrideReadWriteConflict < sc->fftDim / sc->registerBoost + mergeR2C) sc->sharedStrideReadWriteConflict = sc->fftDim / sc->registerBoost + mergeR2C;
		sc->maxSharedStride = (sc->sharedStrideBankConflictFirstStages < sc->sharedStrideReadWriteConflict) ? sc->sharedStrideReadWriteConflict : sc->sharedStrideBankConflictFirstStages;


		sc->usedSharedMemory = vecSize * sc->localSize[1] * sc->maxSharedStride;
		sc->maxSharedStride = ((sc->sharedMemSize < sc->usedSharedMemory)) ? sc->fftDim / sc->registerBoost : sc->maxSharedStride;

		sc->sharedStrideBankConflictFirstStages = (sc->maxSharedStride == sc->fftDim / sc->registerBoost) ? sc->fftDim / sc->registerBoost : sc->sharedStrideBankConflictFirstStages;
		sc->sharedStrideReadWriteConflict = (sc->maxSharedStride == sc->fftDim / sc->registerBoost) ? sc->fftDim / sc->registerBoost : sc->sharedStrideReadWriteConflict;

		sc->tempLen = sprintf(sc->tempStr, "%s sharedStride = %" PRIu64 ";\n", uintType, sc->sharedStrideReadWriteConflict);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;

		sc->tempLen = sprintf(sc->tempStr, "%s* sdata = (%s*)shared;\n\n", vecType, vecType);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		sc->usedSharedMemory = vecSize * sc->localSize[1] * sc->maxSharedStride;
		break;
	}
	case 1: case 2: case 121: case 131: case 141: case 143:
	{
		uint64_t shift = (sc->fftDim < (sc->numSharedBanks / 2)) ? (sc->numSharedBanks / 2) / sc->fftDim : 1;
		sc->sharedStrideReadWriteConflict = ((sc->axisSwapped) && ((sc->localSize[0] % 4) == 0)) ? sc->localSize[0] + shift : sc->localSize[0];
		sc->maxSharedStride = ((maxSequenceSharedMemory < sc->sharedStrideReadWriteConflict* sc->fftDim / sc->registerBoost)) ? sc->localSize[0] : sc->sharedStrideReadWriteConflict;

		sc->sharedStrideReadWriteConflict = (sc->maxSharedStride == sc->localSize[0]) ? sc->localSize[0] : sc->sharedStrideReadWriteConflict;
		sc->tempLen = sprintf(sc->tempStr, "%s sharedStride = %" PRIu64 ";\n", uintType, sc->maxSharedStride);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;


		sc->tempLen = sprintf(sc->tempStr, "%s* sdata = (%s*)shared;\n\n", vecType, vecType);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		sc->usedSharedMemory = vecSize * sc->maxSharedStride * (sc->fftDim + mergeR2C) / sc->registerBoost;

		break;
	}
	}
	return res;
}

FFTResult appendInitialization(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t initType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];


	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");

	uint64_t logicalStoragePerThread = sc->registers_per_thread * sc->registerBoost;
	uint64_t logicalRegistersPerThread = sc->registers_per_thread;

	for (uint64_t i = 0; i < sc->registers_per_thread; i++) {
		sc->tempLen = sprintf(sc->tempStr, "	%s temp_%" PRIu64 ";\n", vecType, i);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	
	//sc->tempLen = sprintf(sc->tempStr, "	uint dum=gl_LocalInvocationID.y;//gl_LocalInvocationID.x/gl_WorkGroupSize.x;\n");
	//sc->tempLen = sprintf(sc->tempStr, "	dum=dum/gl_LocalInvocationID.x-1;\n");
	//sc->tempLen = sprintf(sc->tempStr, "	dummy=dummy/gl_LocalInvocationID.x-1;\n");
	sc->regIDs = (char**)malloc(sizeof(char*) * logicalStoragePerThread);
	if (!sc->regIDs) return FFT_ERROR_MALLOC_FAILED;
	for (uint64_t i = 0; i < logicalStoragePerThread; i++) {
		sc->regIDs[i] = (char*)malloc(sizeof(char) * 50);
		if (!sc->regIDs[i]) {
			for (uint64_t j = 0; j < i; j++) {
				free(sc->regIDs[j]);
				sc->regIDs[j] = 0;
			}
			free(sc->regIDs);
			sc->regIDs = 0;
			return FFT_ERROR_MALLOC_FAILED;
		}
		if (i < logicalRegistersPerThread)
			sprintf(sc->regIDs[i], "temp_%" PRIu64 "", i);
		else
			sprintf(sc->regIDs[i], "temp_%" PRIu64 "", i);
	

	}
	if (sc->registerBoost > 1) {
		
		for (uint64_t i = 1; i < sc->registerBoost; i++) {
			for (uint64_t j = 0; j < sc->registers_per_thread; j++) {
				sc->tempLen = sprintf(sc->tempStr, "	%s temp_%" PRIu64 ";\n", vecType, j + i * sc->registers_per_thread);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}

		}
	}
	sc->tempLen = sprintf(sc->tempStr, "	%s w;\n", vecType);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	sprintf(sc->w, "w");
	uint64_t maxNonPow2Radix = 1;
	if (sc->fftDim % 3 == 0) maxNonPow2Radix = 3;
	if (sc->fftDim % 5 == 0) maxNonPow2Radix = 5;
	if (sc->fftDim % 7 == 0) maxNonPow2Radix = 7;
	if (sc->fftDim % 11 == 0) maxNonPow2Radix = 11;
	if (sc->fftDim % 13 == 0) maxNonPow2Radix = 13;
	for (uint64_t i = 0; i < maxNonPow2Radix; i++) {
		sprintf(sc->locID[i], "loc_%" PRIu64 "", i);
		sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->locID[i]);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	sprintf(sc->temp, "%s", sc->locID[0]);
	uint64_t useRadix8 = 0;
	for (uint64_t i = 0; i < sc->numStages; i++)
		if (sc->stageRadix[i] == 8) useRadix8 = 1;
	if (useRadix8 == 1) {
		if (maxNonPow2Radix > 1) sprintf(sc->iw, "%s", sc->locID[1]);
		else {
			sc->tempLen = sprintf(sc->tempStr, "	%s iw;\n", vecType);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sprintf(sc->iw, "iw");
		}
	}
	//sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", vecType, sc->tempReg);
	sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", uintType, sc->stageInvocationID);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", uintType, sc->blockInvocationID);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", uintType, sc->sdataID);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", uintType, sc->combinedID);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	sc->tempLen = sprintf(sc->tempStr, "	%s %s;\n", uintType, sc->inoutID);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;
	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, "	%s LUTId=0;\n", uintType);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	else {
		sc->tempLen = sprintf(sc->tempStr, "	%s angle=0;\n", floatType);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	if (((sc->stageStartSize > 1) && (!((sc->stageStartSize > 1) && (!sc->reorderFourStep) && (sc->inverse)))) || (((sc->stageStartSize > 1) && (!sc->reorderFourStep) && (sc->inverse))) || (sc->performDCT)) {
		sc->tempLen = sprintf(sc->tempStr, "	%s mult;\n", vecType);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	mult.x = 0;\n");
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		sc->tempLen = sprintf(sc->tempStr, "	mult.y = 0;\n");
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	if (sc->cacheShuffle) {
		sc->tempLen = sprintf(sc->tempStr, "\
	%s tshuffle= ((%s>>1))%%(%" PRIu64 ");\n\
	%s shuffle[%" PRIu64 "];\n", uintType, sc->gl_LocalInvocationID_x, sc->registers_per_thread, vecType, sc->registers_per_thread);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		for (uint64_t i = 0; i < sc->registers_per_thread; i++) {

			sc->tempLen = sprintf(sc->tempStr, "	shuffle[%" PRIu64 "].x = 0;\n", i);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			sc->tempLen = sprintf(sc->tempStr, "	shuffle[%" PRIu64 "].y = 0;\n", i);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}
	}
	return res;
}

FFTResult appendZeropadStart(FFTSpecializationConstantsLayout* sc) {
	//return if sequence is full of zeros from the start
	FFTResult res = FFT_SUCCESS;
	if ((sc->frequencyZeropadding)) {
		switch (sc->axis_id) {
		case 0: {
			break;
		}
		case 1: {
			if (!sc->supportAxis) {
				char idX[500] = "";
				if (sc->performWorkGroupShift[0])
					sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
				else
					sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idX, sc->fft_zeropad_left_full[0], idX, sc->fft_zeropad_right_full[0]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}

			}
			break;
		}
		case 2: {
			if (!sc->supportAxis) {
				char idY[500] = "";
				if (sc->performWorkGroupShift[1])//y axis is along z workgroup here
					sprintf(idY, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
				else
					sprintf(idY, "%s", sc->gl_GlobalInvocationID_z);

				char idX[500] = "";
				if (sc->performWorkGroupShift[0])
					sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
				else
					sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idX, sc->fft_zeropad_left_full[0], idX, sc->fft_zeropad_right_full[0]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idY, sc->fft_zeropad_left_full[1], idY, sc->fft_zeropad_right_full[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			else {
				char idY[500] = "";
				if (sc->performWorkGroupShift[1])//for support axes y is along x workgroup
					sprintf(idY, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
				else
					sprintf(idY, "%s", sc->gl_GlobalInvocationID_x);
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idY, sc->fft_zeropad_left_full[1], idY, sc->fft_zeropad_right_full[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			break;
		}
		}
	}
	else {
		switch (sc->axis_id) {
		case 0: {
			char idY[500] = "";
			if (sc->axisSwapped) {
				if (sc->performWorkGroupShift[1])
					sprintf(idY, "(%s + (%s + consts.workGroupShiftY) * %" PRIu64 ")", sc->gl_LocalInvocationID_x, sc->gl_WorkGroupID_y, sc->localSize[0]);
				else
					sprintf(idY, "%s + %s * %" PRIu64 "", sc->gl_LocalInvocationID_x, sc->gl_WorkGroupID_y, sc->localSize[0]);

				char idZ[500] = "";
				if (sc->performWorkGroupShift[2])
					sprintf(idZ, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
				else
					sprintf(idZ, "%s", sc->gl_GlobalInvocationID_z);
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idY, sc->fft_zeropad_left_full[1], idY, sc->fft_zeropad_right_full[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				if (sc->performZeropaddingFull[2]) {
					if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idZ, sc->fft_zeropad_left_full[2], idZ, sc->fft_zeropad_right_full[2]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			else {
				if (sc->performWorkGroupShift[1])
					sprintf(idY, "(%s + consts.workGroupShiftY * %s)", sc->gl_GlobalInvocationID_y, sc->gl_WorkGroupSize_y);
				else
					sprintf(idY, "%s", sc->gl_GlobalInvocationID_y);

				char idZ[500] = "";
				if (sc->performWorkGroupShift[2])
					sprintf(idZ, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
				else
					sprintf(idZ, "%s", sc->gl_GlobalInvocationID_z);
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idY, sc->fft_zeropad_left_full[1], idY, sc->fft_zeropad_right_full[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				if (sc->performZeropaddingFull[2]) {
					if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idZ, sc->fft_zeropad_left_full[2], idZ, sc->fft_zeropad_right_full[2]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			break;
		}
		case 1: {
			char idZ[500] = "";
			if (sc->performWorkGroupShift[2])
				sprintf(idZ, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
			else
				sprintf(idZ, "%s", sc->gl_GlobalInvocationID_z);
			if (sc->performZeropaddingFull[2]) {
				if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2]) {
					sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idZ, sc->fft_zeropad_left_full[2], idZ, sc->fft_zeropad_right_full[2]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}

			break;
		}
		case 2: {

			break;
		}
		}
	}
	return res;
}

FFTResult appendZeropadEnd(FFTSpecializationConstantsLayout* sc) {
	//return if sequence is full of zeros from the start
	FFTResult res = FFT_SUCCESS;
	if ((sc->frequencyZeropadding)) {
		switch (sc->axis_id) {
		case 0: {
			break;
		}
		case 1: {
			if (!sc->supportAxis) {
				char idX[500] = "";
				if (sc->performWorkGroupShift[0])
					sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
				else
					sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0]) {
						sc->tempLen = sprintf(sc->tempStr, "		}\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}

			}
			break;
		}
		case 2: {
			if (!sc->supportAxis) {
				char idY[500] = "";
				if (sc->performWorkGroupShift[1])//y axis is along z workgroup here
					sprintf(idY, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
				else
					sprintf(idY, "%s", sc->gl_GlobalInvocationID_z);

				char idX[500] = "";
				if (sc->performWorkGroupShift[0])
					sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
				else
					sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0]) {
						sc->tempLen = sprintf(sc->tempStr, "		}\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
						sc->tempLen = sprintf(sc->tempStr, "		}\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			else {
				char idY[500] = "";
				if (sc->performWorkGroupShift[1])//for support axes y is along x workgroup
					sprintf(idY, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
				else
					sprintf(idY, "%s", sc->gl_GlobalInvocationID_x);
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
						sc->tempLen = sprintf(sc->tempStr, "		}\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			break;
		}
		}
	}
	else {
		switch (sc->axis_id) {
		case 0: {
			char idY[500] = "";
			if (sc->performZeropaddingFull[1]) {
				if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			if (sc->performZeropaddingFull[2]) {
				if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2]) {
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			break;
		}
		case 1: {
			char idZ[500] = "";
			if (sc->performWorkGroupShift[2])
				sprintf(idZ, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
			else
				sprintf(idZ, "%s", sc->gl_GlobalInvocationID_z);
			if (sc->performZeropaddingFull[2]) {
				if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2]) {
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			break;
		}
		case 2: {

			break;
		}
		}
	}
	return res;
}


FFTResult appendBarrierFFT(FFTSpecializationConstantsLayout* sc, uint64_t numTab) {
	FFTResult res = FFT_SUCCESS;
	char tabs[100];
	for (uint64_t i = 0; i < numTab; i++)
		sprintf(tabs, "	");

	sc->tempLen = sprintf(sc->tempStr, "%s__syncthreads();\n\n", tabs);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) return res;

	return res;
}


FFTResult appendBoostThreadDataReorder(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t shuffleType, uint64_t start) {
	FFTResult res = FFT_SUCCESS;
	switch (shuffleType) {
	case 0: case 5: case 6: case 120: case 130: case 140: case 142: {
		uint64_t logicalStoragePerThread;
		if (start == 1) {
			logicalStoragePerThread = sc->registers_per_thread_per_radix[sc->stageRadix[0]] * sc->registerBoost;// (sc->registers_per_thread % sc->stageRadix[0] == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
		}
		else {
			logicalStoragePerThread = sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] * sc->registerBoost;// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
		}
		uint64_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
		if ((sc->registerBoost > 1) && (logicalStoragePerThread != sc->min_registers_per_thread * sc->registerBoost)) {
			for (uint64_t k = 0; k < sc->registerBoost; k++) {
				if (k > 0) {
					res = appendBarrierFFT(sc, 2);
					if (res != FFT_SUCCESS) return res;
				}
				res = appendZeropadStart(sc);
				if (res != FFT_SUCCESS) return res;
				res = AppendLineFromInput(sc, sc->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (start == 0) {
					sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThread, sc->fftDim);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					for (uint64_t i = 0; i < logicalStoragePerThread / sc->registerBoost; i++) {
						sc->tempLen = sprintf(sc->tempStr, "\
	sdata[%s + %" PRIu64 "] = %s;\n", sc->gl_LocalInvocationID_x, i * logicalGroupSize, sc->regIDs[i + k * sc->registers_per_thread]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "	}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else
				{
					for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
						sc->tempLen = sprintf(sc->tempStr, "\
	sdata[%s + %" PRIu64 "] = %s;\n", sc->gl_LocalInvocationID_x, i * sc->localSize[0], sc->regIDs[i + k * sc->registers_per_thread]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				res = AppendLineFromInput(sc, sc->disableThreadsEnd);
				if (res != FFT_SUCCESS) return res;
				res = appendZeropadEnd(sc);
				if (res != FFT_SUCCESS) return res;
				res = appendBarrierFFT(sc, 2);
				if (res != FFT_SUCCESS) return res;
				res = appendZeropadStart(sc);
				if (res != FFT_SUCCESS) return res;
				res = AppendLineFromInput(sc, sc->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (start == 1) {
					sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThread, sc->fftDim);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					for (uint64_t i = 0; i < logicalStoragePerThread / sc->registerBoost; i++) {
						sc->tempLen = sprintf(sc->tempStr, "\
	%s = sdata[%s + %" PRIu64 "];\n", sc->regIDs[i + k * sc->registers_per_thread], sc->gl_LocalInvocationID_x, i * logicalGroupSize);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "	}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
						sc->tempLen = sprintf(sc->tempStr, "\
	%s = sdata[%s + %" PRIu64 "];\n", sc->regIDs[i + k * sc->registers_per_thread], sc->gl_LocalInvocationID_x, i * sc->localSize[0]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				res = AppendLineFromInput(sc, sc->disableThreadsEnd);
				if (res != FFT_SUCCESS) return res;
				res = appendZeropadEnd(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}

		break;
	}
	case 1: case 2: case 121: case 131: case 141: case 143: {
		uint64_t logicalStoragePerThread;
		if (start == 1) {
			logicalStoragePerThread = sc->registers_per_thread_per_radix[sc->stageRadix[0]] * sc->registerBoost;// (sc->registers_per_thread % sc->stageRadix[0] == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
		}
		else {
			logicalStoragePerThread = sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] * sc->registerBoost;// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
		}
		uint64_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
		if ((sc->registerBoost > 1) && (logicalStoragePerThread != sc->min_registers_per_thread * sc->registerBoost)) {
			for (uint64_t k = 0; k < sc->registerBoost; k++) {
				if (k > 0) {
					res = appendBarrierFFT(sc, 2);
					if (res != FFT_SUCCESS) return res;
				}
				res = appendZeropadStart(sc);
				if (res != FFT_SUCCESS) return res;
				res = AppendLineFromInput(sc, sc->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (start == 0) {
					sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_y, logicalStoragePerThread, sc->fftDim);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					for (uint64_t i = 0; i < logicalStoragePerThread / sc->registerBoost; i++) {
						sc->tempLen = sprintf(sc->tempStr, "\
	sdata[%s + %s * (%s + %" PRIu64 ")] = %s;\n", sc->gl_LocalInvocationID_x, sc->sharedStride, sc->gl_LocalInvocationID_y, i * logicalGroupSize, sc->regIDs[i + k * sc->registers_per_thread]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "	}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else
				{
					for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
						sc->tempLen = sprintf(sc->tempStr, "\
	sdata[%s + %s * (%s + %" PRIu64 ")] = %s;\n", sc->gl_LocalInvocationID_x, sc->sharedStride, sc->gl_LocalInvocationID_y, i * sc->localSize[1], sc->regIDs[i + k * sc->registers_per_thread]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				res = AppendLineFromInput(sc, sc->disableThreadsEnd);
				if (res != FFT_SUCCESS) return res;
				res = appendZeropadEnd(sc);
				if (res != FFT_SUCCESS) return res;
				res = appendBarrierFFT(sc, 2);
				if (res != FFT_SUCCESS) return res;
				res = appendZeropadStart(sc);
				if (res != FFT_SUCCESS) return res;
				res = AppendLineFromInput(sc, sc->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (start == 1) {
					sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_y, logicalStoragePerThread, sc->fftDim);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					for (uint64_t i = 0; i < logicalStoragePerThread / sc->registerBoost; i++) {
						sc->tempLen = sprintf(sc->tempStr, "\
	%s = sdata[%s + %s * (%s + %" PRIu64 ")];\n", sc->regIDs[i + k * sc->registers_per_thread], sc->gl_LocalInvocationID_x, sc->sharedStride, sc->gl_LocalInvocationID_y, i * logicalGroupSize);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "	}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
						sc->tempLen = sprintf(sc->tempStr, "\
	%s = sdata[%s + %s * (%s + %" PRIu64 ")];\n", sc->regIDs[i + k * sc->registers_per_thread], sc->gl_LocalInvocationID_x, sc->sharedStride, sc->gl_LocalInvocationID_y, i * sc->localSize[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				res = AppendLineFromInput(sc, sc->disableThreadsEnd);
				if (res != FFT_SUCCESS) return res;
				res = appendZeropadEnd(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}

		break;
	}
	}
	return res;
}



FFTResult appendRadixStageNonStrided(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char LFending[4] = "";
	if (!strcmp(floatType, "float")) sprintf(LFending, "f");

	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");


	char convolutionInverse[10] = "";
	if (sc->convolutionStep) {
		if (stageAngle < 0)
			sprintf(convolutionInverse, ", 0");
		else
			sprintf(convolutionInverse, ", 1");
	}
	uint64_t logicalStoragePerThread = sc->registers_per_thread_per_radix[stageRadix] * sc->registerBoost;// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	uint64_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[stageRadix];// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	uint64_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
	if ((sc->localSize[0] * logicalStoragePerThread > sc->fftDim) || (stageSize > 1) || ((sc->localSize[1] > 1) && (!(sc->performR2C && (stageAngle > 0)))) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle > 0)) || (sc->performDCT))
	{
		res = appendBarrierFFT(sc, 1);
		if (res != FFT_SUCCESS) return res;
	}
	res = appendZeropadStart(sc);
	if (res != FFT_SUCCESS) return res;
	res = AppendLineFromInput(sc, sc->disableThreadsStart);
	if (res != FFT_SUCCESS) return res;

	if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim) {
		sc->tempLen = sprintf(sc->tempStr, "\
		if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_LocalInvocationID_x, logicalStoragePerThread, sc->fftDim);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	for (uint64_t k = 0; k < sc->registerBoost; k++) {
		for (uint64_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
			sc->tempLen = sprintf(sc->tempStr, "\
		%s = (%s+ %" PRIu64 ") %% (%" PRIu64 ");\n", sc->stageInvocationID, sc->gl_LocalInvocationID_x, (j + k * logicalRegistersPerThread / stageRadix) * logicalGroupSize, stageSize);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if (sc->LUT)
				sc->tempLen = sprintf(sc->tempStr, "		LUTId = stageInvocationID + %" PRIu64 ";\n", stageSizeSum);
			else
				sc->tempLen = sprintf(sc->tempStr, "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if ((sc->registerBoost == 1) && ((sc->localSize[0] * logicalStoragePerThread > sc->fftDim) || (stageSize > 1) || ((sc->localSize[1] > 1) && (!(sc->performR2C && (stageAngle > 0)))) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle > 0)) || sc->performDCT)) {
				for (uint64_t i = 0; i < stageRadix; i++) {
					uint64_t id = j + i * logicalRegistersPerThread / stageRadix;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;

					sc->tempLen = sprintf(sc->tempStr, "\
		%s = %s + %" PRIu64 ";\n", sc->sdataID, sc->gl_LocalInvocationID_x, j * logicalGroupSize + i * sc->fftDim / stageRadix);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					if (sc->resolveBankConflictFirstStages == 1) {
						sc->tempLen = sprintf(sc->tempStr, "\
	%s = (%s / %" PRIu64 ") * %" PRIu64 " + %s %% %" PRIu64 ";", sc->sdataID, sc->sdataID, sc->numSharedBanks / 2, sc->numSharedBanks / 2 + 1, sc->sdataID, sc->numSharedBanks / 2);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}

					if (sc->localSize[1] > 1) {
						sc->tempLen = sprintf(sc->tempStr, "\
		%s = %s + sharedStride * %s;\n", sc->sdataID, sc->sdataID, sc->gl_LocalInvocationID_y);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "\
		%s = sdata[%s];\n", sc->regIDs[id], sc->sdataID);
					res = AppendLine(sc);
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
					uint64_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sprintf(regID[i], "%s", sc->regIDs[id]);
					/*if(j + i * logicalStoragePerThread / stageRadix < logicalRegistersPerThread)
						sprintf(regID[i], "%s", sc->regIDs[j + i * logicalStoragePerThread / stageRadix]);
					else
						sprintf(regID[i], "%" PRIu64 "[%" PRIu64 "]", (j + i * logicalStoragePerThread / stageRadix)/ logicalRegistersPerThread, (j + i * logicalStoragePerThread / stageRadix) % logicalRegistersPerThread);*/

				}
				res = inlineRadixKernelFFT(sc, floatType, uintType, stageRadix, stageSize, stageAngle, regID);
				if (res != FFT_SUCCESS) return res;
				for (uint64_t i = 0; i < stageRadix; i++) {
					uint64_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sprintf(sc->regIDs[id], "%s", regID[i]);
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
		if ((stageSize == 1) && (sc->cacheShuffle)) {
			for (uint64_t i = 0; i < logicalRegistersPerThread; i++) {
				uint64_t id = i + k * logicalRegistersPerThread;
				id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
				sc->tempLen = sprintf(sc->tempStr, "\
		shuffle[%" PRIu64 "]=%s;\n", i, sc->regIDs[id]);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
			for (uint64_t i = 0; i < logicalRegistersPerThread; i++) {
				uint64_t id = i + k * logicalRegistersPerThread;
				id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
				sc->tempLen = sprintf(sc->tempStr, "\
		%s=shuffle[(%" PRIu64 "+tshuffle)%%(%" PRIu64 ")];\n", sc->regIDs[id], i, logicalRegistersPerThread);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
	}
	if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim) {
		sc->tempLen = sprintf(sc->tempStr, "		}\n");
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	res = AppendLineFromInput(sc, sc->disableThreadsEnd);
	if (res != FFT_SUCCESS) return res;
	res = appendZeropadEnd(sc);
	if (res != FFT_SUCCESS) return res;
	return res;
}
FFTResult appendRadixStageStrided(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char LFending[4] = "";
	if (!strcmp(floatType, "float")) sprintf(LFending, "f");

	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");


	char convolutionInverse[10] = "";
	if (sc->convolutionStep) {
		if (stageAngle < 0)
			sprintf(convolutionInverse, ", 0");
		else
			sprintf(convolutionInverse, ", 1");
	}
	uint64_t logicalStoragePerThread = sc->registers_per_thread_per_radix[stageRadix] * sc->registerBoost;// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	uint64_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[stageRadix];// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	uint64_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
	if (((sc->axis_id == 0) && (sc->axis_upload_id == 0) && (!(sc->performR2C && (stageAngle > 0)))) || (sc->localSize[1] * logicalStoragePerThread > sc->fftDim) || (stageSize > 1) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle > 0)) || (sc->performDCT))
	{
		res = appendBarrierFFT(sc, 1);
		if (res != FFT_SUCCESS) return res;
	}
	res = appendZeropadStart(sc);
	if (res != FFT_SUCCESS) return res;
	res = AppendLineFromInput(sc, sc->disableThreadsStart);
	if (res != FFT_SUCCESS) return res;
	if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim) {
		sc->tempLen = sprintf(sc->tempStr, "\
		if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_LocalInvocationID_y, logicalStoragePerThread, sc->fftDim);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	for (uint64_t k = 0; k < sc->registerBoost; k++) {
		for (uint64_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
			sc->tempLen = sprintf(sc->tempStr, "\
		%s = (%s+ %" PRIu64 ") %% (%" PRIu64 ");\n", sc->stageInvocationID, sc->gl_LocalInvocationID_y, (j + k * logicalRegistersPerThread / stageRadix) * logicalGroupSize, stageSize);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if (sc->LUT)
				sc->tempLen = sprintf(sc->tempStr, "		LUTId = stageInvocationID + %" PRIu64 ";\n", stageSizeSum);
			else
				sc->tempLen = sprintf(sc->tempStr, "		angle = stageInvocationID * %.17f%s;\n", stageAngle, LFending);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
			if ((sc->registerBoost == 1) && (((sc->axis_id == 0) && (sc->axis_upload_id == 0) && (!(sc->performR2C && (stageAngle > 0)))) || (sc->localSize[1] * logicalStoragePerThread > sc->fftDim) || (stageSize > 1) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle > 0)) || (sc->performDCT))) {
				for (uint64_t i = 0; i < stageRadix; i++) {
					uint64_t id = j + i * logicalRegistersPerThread / stageRadix;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sc->tempLen = sprintf(sc->tempStr, "\
		%s = sdata[%s*(%s+%" PRIu64 ")+%s];\n", sc->regIDs[id], sc->sharedStride, sc->gl_LocalInvocationID_y, j * logicalGroupSize + i * sc->fftDim / stageRadix, sc->gl_LocalInvocationID_x);
					res = AppendLine(sc);
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
					uint64_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sprintf(regID[i], "%s", sc->regIDs[id]);
					/*if (j + i * logicalStoragePerThread / stageRadix < logicalRegistersPerThread)
						sprintf(regID[i], "_%" PRIu64 "", j + i * logicalStoragePerThread / stageRadix);
					else
						sprintf(regID[i], "%" PRIu64 "[%" PRIu64 "]", (j + i * logicalStoragePerThread / stageRadix) / logicalRegistersPerThread, (j + i * logicalStoragePerThread / stageRadix) % logicalRegistersPerThread);*/

				}
				res = inlineRadixKernelFFT(sc, floatType, uintType, stageRadix, stageSize, stageAngle, regID);
				if (res != FFT_SUCCESS) return res;
				for (uint64_t i = 0; i < stageRadix; i++) {
					uint64_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
					id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
					sprintf(sc->regIDs[id], "%s", regID[i]);
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
	if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim) {
		sc->tempLen = sprintf(sc->tempStr, "		}\n");
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	res = AppendLineFromInput(sc, sc->disableThreadsEnd);
	if (res != FFT_SUCCESS) return res;
	res = appendZeropadEnd(sc);
	if (res != FFT_SUCCESS) return res;
	if (stageSize == 1) {
		sc->tempLen = sprintf(sc->tempStr, "		%s = %" PRIu64 ";\n", sc->sharedStride, sc->localSize[0]);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	return res;
}


FFTResult appendRadixStage(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t shuffleType) {
	FFTResult res = FFT_SUCCESS;
	switch (shuffleType) {
	case 0: case 5: case 6: case 120: case 130: case 140: case 142: {
		res = appendRadixStageNonStrided(sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 1: case 2: case 121: case 131: case 141: case 143: {
		res = appendRadixStageStrided(sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	}
	return res;
}





FFTResult appendRegisterBoostShuffle(FFTSpecializationConstantsLayout* sc, const char* floatType, uint64_t stageSize, uint64_t stageRadixPrev, uint64_t stageRadix, double stageAngle) {
	FFTResult res = FFT_SUCCESS;
	if (((sc->actualInverse) && (sc->normalize)) || ((sc->convolutionStep) && (stageAngle > 0))) {
		char stageNormalization[10] = "";
		if ((stageSize == 1) && (sc->performDCT)) {
			if (sc->performDCT == 4)
				sprintf(stageNormalization, "%" PRIu64 "", stageRadixPrev * stageRadix * 4);
			else
				sprintf(stageNormalization, "%" PRIu64 "", stageRadixPrev * stageRadix * 2);
		}
		else
			sprintf(stageNormalization, "%" PRIu64 "", stageRadixPrev * stageRadix);
		uint64_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[stageRadix];// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
		for (uint64_t k = 0; k < sc->registerBoost; ++k) {
			for (uint64_t i = 0; i < logicalRegistersPerThread; i++) {
				res = DivComplexNumber(sc, sc->regIDs[i + k * sc->registers_per_thread], sc->regIDs[i + k * sc->registers_per_thread], stageNormalization);
				if (res != FFT_SUCCESS) return res;
			}
		}
	}
	return res;
}




FFTResult appendRadixShuffleNonStrided(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];

	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");

	char stageNormalization[10] = "";
	if (((sc->actualInverse) && (sc->normalize)) || ((sc->convolutionStep) && (stageAngle > 0))) {
		if ((stageSize == 1) && (sc->performDCT)) {
			if (sc->performDCT == 4)
				sprintf(stageNormalization, "%" PRIu64 "", stageRadix * 4);
			else
				sprintf(stageNormalization, "%" PRIu64 "", stageRadix * 2);
		}
		else
			sprintf(stageNormalization, "%" PRIu64 "", stageRadix);
	}

	char tempNum[50] = "";

	uint64_t logicalStoragePerThread = sc->registers_per_thread_per_radix[stageRadix] * sc->registerBoost;// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	uint64_t logicalStoragePerThreadNext = sc->registers_per_thread_per_radix[stageRadixNext] * sc->registerBoost;// (sc->registers_per_thread % stageRadixNext == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	uint64_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[stageRadix];// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	uint64_t logicalRegistersPerThreadNext = sc->registers_per_thread_per_radix[stageRadixNext];// (sc->registers_per_thread % stageRadixNext == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;

	uint64_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
	uint64_t logicalGroupSizeNext = sc->fftDim / logicalStoragePerThreadNext;
	if (((sc->registerBoost == 1) && ((sc->localSize[0] * logicalStoragePerThread > sc->fftDim) || (stageSize < sc->fftDim / stageRadix) || ((sc->reorderFourStep) && (sc->fftDim < sc->fft_dim_full) && (sc->localSize[1] > 1)) || (sc->localSize[1] > 1) || ((sc->performR2C) && (!sc->inverse) && (sc->axis_id == 0)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle < 0)))) || (sc->performDCT))
	{
		res = appendBarrierFFT(sc, 1);
		if (res != FFT_SUCCESS) return res;
	}
	if ((sc->localSize[0] * logicalStoragePerThread > sc->fftDim) || (stageSize < sc->fftDim / stageRadix) || ((sc->reorderFourStep) && (sc->fftDim < sc->fft_dim_full) && (sc->localSize[1] > 1)) || (sc->localSize[1] > 1) || ((sc->performR2C) && (!sc->inverse) && (sc->axis_id == 0)) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle < 0)) || (sc->registerBoost > 1) || (sc->performDCT)) {
		//appendBarrierFFT(sc, 1);
		if (!((sc->registerBoost > 1) && (stageSize * stageRadix == sc->fftDim / sc->stageRadix[sc->numStages - 1]) && (sc->stageRadix[sc->numStages - 1] == sc->registerBoost))) {
			char** tempID;
			tempID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
			if (tempID) {
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
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
				res = appendZeropadStart(sc);
				if (res != FFT_SUCCESS) return res;
				res = AppendLineFromInput(sc, sc->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim) {
					sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThread, sc->fftDim);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				for (uint64_t k = 0; k < sc->registerBoost; ++k) {
					uint64_t t = 0;
					if (k > 0) {
						res = appendBarrierFFT(sc, 2);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadStart(sc);
						if (res != FFT_SUCCESS) return res;
						res = AppendLineFromInput(sc, sc->disableThreadsStart);
						if (res != FFT_SUCCESS) return res;
						if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim) {
							sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThread, sc->fftDim);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
					for (uint64_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
						sprintf(tempNum, "%" PRIu64 "", j * logicalGroupSize);
						res = AddReal(sc, sc->stageInvocationID, sc->gl_LocalInvocationID_x, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = MovReal(sc, sc->blockInvocationID, sc->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						sprintf(tempNum, "%" PRIu64 "", stageSize);
						res = ModReal(sc, sc->stageInvocationID, sc->stageInvocationID, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = SubReal(sc, sc->blockInvocationID, sc->blockInvocationID, sc->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						sprintf(tempNum, "%" PRIu64 "", stageRadix);
						res = MulReal(sc, sc->inoutID, sc->blockInvocationID, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = AddReal(sc, sc->inoutID, sc->inoutID, sc->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						
						if ((stageSize == 1) && (sc->cacheShuffle)) {
							for (uint64_t i = 0; i < stageRadix; i++) {
								uint64_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
								id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
								sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
								t++;
								sprintf(tempNum, "%" PRIu64 "", i);
								res = AddReal(sc, sc->sdataID, tempNum, sc->tshuffle);
								if (res != FFT_SUCCESS) return res;
								sprintf(tempNum, "%" PRIu64 "", logicalRegistersPerThread);
								res = ModReal(sc, sc->sdataID, sc->sdataID, tempNum);
								if (res != FFT_SUCCESS) return res;
								sprintf(tempNum, "%" PRIu64 "", stageSize);
								res = MulReal(sc, sc->sdataID, sc->sdataID, tempNum);
								if (res != FFT_SUCCESS) return res;
								if (sc->localSize[1] > 1) {
									res = MulReal(sc, sc->combinedID, sc->gl_LocalInvocationID_y, sc->sharedStride);
									if (res != FFT_SUCCESS) return res;
									res = AddReal(sc, sc->sdataID, sc->sdataID, sc->combinedID);
									if (res != FFT_SUCCESS) return res;
								}
								res = AddReal(sc, sc->sdataID, sc->sdataID, sc->inoutID);
								if (res != FFT_SUCCESS) return res;

								//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + inoutID + ((%" PRIu64 "+tshuffle) %% (%" PRIu64 "))*%" PRIu64 "", i, logicalRegistersPerThread, stageSize);
								if (strcmp(stageNormalization, "")) {
									res = DivComplexNumber(sc, sc->regIDs[id], sc->regIDs[id], stageNormalization);
									if (res != FFT_SUCCESS) return res;
								}
								res = SharedStore(sc, sc->sdataID, sc->regIDs[id]);
								if (res != FFT_SUCCESS) return res;
								
							}
						}
						else {
							for (uint64_t i = 0; i < stageRadix; i++) {
								uint64_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
								id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
								sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
								t++;
								sprintf(tempNum, "%" PRIu64 "", i * stageSize);
								res = AddReal(sc, sc->sdataID, sc->inoutID, tempNum);
								if (res != FFT_SUCCESS) return res;
								if ((stageSize <= sc->numSharedBanks / 2) && (sc->fftDim > sc->numSharedBanks / 2) && (sc->sharedStrideBankConflictFirstStages != sc->fftDim / sc->registerBoost) && ((sc->fftDim & (sc->fftDim - 1)) == 0) && (stageSize * stageRadix != sc->fftDim)) {
									if (sc->resolveBankConflictFirstStages == 0) {
										sc->resolveBankConflictFirstStages = 1;
										sc->tempLen = sprintf(sc->tempStr, "\
	%s = %" PRIu64 ";", sc->sharedStride, sc->sharedStrideBankConflictFirstStages);
										res = AppendLine(sc);
										if (res != FFT_SUCCESS) return res;
									}
									sc->tempLen = sprintf(sc->tempStr, "\
	%s = (%s / %" PRIu64 ") * %" PRIu64 " + %s %% %" PRIu64 ";", sc->sdataID, sc->sdataID, sc->numSharedBanks / 2, sc->numSharedBanks / 2 + 1, sc->sdataID, sc->numSharedBanks / 2);
									res = AppendLine(sc);
									if (res != FFT_SUCCESS) return res;

								}
								else {
									if (sc->resolveBankConflictFirstStages == 1) {
										sc->resolveBankConflictFirstStages = 0;
										sc->tempLen = sprintf(sc->tempStr, "\
	%s = %" PRIu64 ";", sc->sharedStride, sc->sharedStrideReadWriteConflict);
										res = AppendLine(sc);
										if (res != FFT_SUCCESS) return res;
									}
								}
								if (sc->localSize[1] > 1) {
									res = MulReal(sc, sc->combinedID, sc->gl_LocalInvocationID_y, sc->sharedStride);
									if (res != FFT_SUCCESS) return res;
									res = AddReal(sc, sc->sdataID, sc->sdataID, sc->combinedID);
									if (res != FFT_SUCCESS) return res;
								}
								//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + inoutID + %" PRIu64 "", i * stageSize);
								if (strcmp(stageNormalization, "")) {
									res = DivComplexNumber(sc, sc->regIDs[id], sc->regIDs[id], stageNormalization);
									if (res != FFT_SUCCESS) return res;
								}
								res = SharedStore(sc, sc->sdataID, sc->regIDs[id]);
								if (res != FFT_SUCCESS) return res;
								/*sc->tempLen = sprintf(sc->tempStr, "\
	sdata[sharedStride * gl_LocalInvocationID.y + inoutID + %" PRIu64 "] = temp%s%s;\n", i * stageSize, sc->regIDs[id], stageNormalization);*/
							}
						}
					}
					for (uint64_t j = logicalRegistersPerThread; j < sc->registers_per_thread; j++) {
						sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[t + k * sc->registers_per_thread]);
						t++;
					}
					t = 0;
					if (sc->registerBoost > 1) {
						if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
						{
							sc->tempLen = sprintf(sc->tempStr, "	}\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(sc, sc->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadEnd(sc);
						if (res != FFT_SUCCESS) return res;
						res = appendBarrierFFT(sc, 2);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadStart(sc);
						if (res != FFT_SUCCESS) return res;
						res = AppendLineFromInput(sc, sc->disableThreadsStart);
						if (res != FFT_SUCCESS) return res;
						if (sc->localSize[0] * logicalStoragePerThreadNext > sc->fftDim) {
							sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThreadNext, sc->fftDim);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						for (uint64_t j = 0; j < logicalRegistersPerThreadNext / stageRadixNext; j++) {
							for (uint64_t i = 0; i < stageRadixNext; i++) {
								uint64_t id = j + k * logicalRegistersPerThreadNext / stageRadixNext + i * logicalStoragePerThreadNext / stageRadixNext;
								id = (id / logicalRegistersPerThreadNext) * sc->registers_per_thread + id % logicalRegistersPerThreadNext;
								//resID[t + k * sc->registers_per_thread] = sc->regIDs[id];
								sprintf(tempNum, "%" PRIu64 "", t * logicalGroupSizeNext);
								res = AddReal(sc, sc->sdataID, sc->gl_LocalInvocationID_x, tempNum);
								if (res != FFT_SUCCESS) return res;
								if (sc->localSize[1] > 1) {
									res = MulReal(sc, sc->combinedID, sc->gl_LocalInvocationID_y, sc->sharedStride);
									if (res != FFT_SUCCESS) return res;
									res = AddReal(sc, sc->sdataID, sc->sdataID, sc->combinedID);
									if (res != FFT_SUCCESS) return res;
								}
								res = SharedLoad(sc, tempID[t + k * sc->registers_per_thread], sc->sdataID);
								if (res != FFT_SUCCESS) return res;
								
								t++;
							}

						}
						if (sc->localSize[0] * logicalStoragePerThreadNext > sc->fftDim)
						{
							sc->tempLen = sprintf(sc->tempStr, "	}\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(sc, sc->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadEnd(sc);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
						{
							sc->tempLen = sprintf(sc->tempStr, "	}\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(sc, sc->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadEnd(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					//printf("0 - %s\n", resID[i]);
					sprintf(sc->regIDs[i], "%s", tempID[i]);
					//sprintf(resID[i], "%s", tempID[i]);
					//printf("1 - %s\n", resID[i]);
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
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
			tempID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
			if (tempID) {
				//resID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
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
				for (uint64_t k = 0; k < sc->registerBoost; ++k) {
					for (uint64_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
						for (uint64_t i = 0; i < stageRadix; i++) {
							uint64_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
							id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
							sprintf(tempID[j + i * logicalRegistersPerThread / stageRadix + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
						}
					}
					for (uint64_t j = logicalRegistersPerThread; j < sc->registers_per_thread; j++) {
						sprintf(tempID[j + k * sc->registers_per_thread], "%s", sc->regIDs[j + k * sc->registers_per_thread]);
					}
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					sprintf(sc->regIDs[i], "%s", tempID[i]);
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
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
		res = appendZeropadStart(sc);
		if (res != FFT_SUCCESS) return res;
		res = AppendLineFromInput(sc, sc->disableThreadsStart);
		if (res != FFT_SUCCESS) return res;
		if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim) {
			sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, logicalStoragePerThread, sc->fftDim);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}
		if (((sc->actualInverse) && (sc->normalize)) || ((sc->convolutionStep) && (stageAngle > 0))) {
			for (uint64_t i = 0; i < logicalStoragePerThread; i++) {
				res = DivComplexNumber(sc, sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], stageNormalization);
				if (res != FFT_SUCCESS) return res;
				/*sc->tempLen = sprintf(sc->tempStr, "\
	temp%s = temp%s%s;\n", sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], stageNormalization);*/
			}
		}
		if (sc->localSize[0] * logicalStoragePerThread > sc->fftDim)
		{
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}
		res = AppendLineFromInput(sc, sc->disableThreadsEnd);
		if (res != FFT_SUCCESS) return res;
		res = appendZeropadEnd(sc);
		if (res != FFT_SUCCESS) return res;
	}
	return res;
}


FFTResult appendRadixShuffleStrided(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];

	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");


	char stageNormalization[10] = "";
	char tempNum[50] = "";

	uint64_t logicalStoragePerThread = sc->registers_per_thread_per_radix[stageRadix] * sc->registerBoost;// (sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	uint64_t logicalStoragePerThreadNext = sc->registers_per_thread_per_radix[stageRadixNext] * sc->registerBoost;//(sc->registers_per_thread % stageRadixNext == 0) ? sc->registers_per_thread * sc->registerBoost : sc->min_registers_per_thread * sc->registerBoost;
	uint64_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[stageRadix];//(sc->registers_per_thread % stageRadix == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	uint64_t logicalRegistersPerThreadNext = sc->registers_per_thread_per_radix[stageRadixNext];//(sc->registers_per_thread % stageRadixNext == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;

	uint64_t logicalGroupSize = sc->fftDim / logicalStoragePerThread;
	uint64_t logicalGroupSizeNext = sc->fftDim / logicalStoragePerThreadNext;
	if (((sc->actualInverse) && (sc->normalize)) || ((sc->convolutionStep) && (stageAngle > 0))) {
		if ((stageSize == 1) && (sc->performDCT)) {
			if (sc->performDCT == 4)
				sprintf(stageNormalization, "%" PRIu64 "", stageRadix * 4);
			else
				sprintf(stageNormalization, "%" PRIu64 "", stageRadix * 2);
		}
		else
			sprintf(stageNormalization, "%" PRIu64 "", stageRadix);
	}
	if (((sc->axis_id == 0) && (sc->axis_upload_id == 0)) || (sc->localSize[1] * logicalStoragePerThread > sc->fftDim) || (stageSize < sc->fftDim / stageRadix) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle < 0)) || (sc->performDCT))
	{
		res = appendBarrierFFT(sc, 2);
		if (res != FFT_SUCCESS) return res;
	}
	if (stageSize == sc->fftDim / stageRadix) {
		sc->tempLen = sprintf(sc->tempStr, "		%s = %" PRIu64 ";\n", sc->sharedStride, sc->sharedStrideReadWriteConflict);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
	}
	if (((sc->axis_id == 0) && (sc->axis_upload_id == 0)) || (sc->localSize[1] * logicalStoragePerThread > sc->fftDim) || (stageSize < sc->fftDim / stageRadix) || ((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels > 1)) && (stageAngle < 0)) || (sc->performDCT)) {
		//appendBarrierFFT(sc, 2);
		if (!((sc->registerBoost > 1) && (stageSize * stageRadix == sc->fftDim / sc->stageRadix[sc->numStages - 1]) && (sc->stageRadix[sc->numStages - 1] == sc->registerBoost))) {
			char** tempID;
			tempID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
			if (tempID) {
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
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
				res = appendZeropadStart(sc);
				if (res != FFT_SUCCESS) return res;
				res = AppendLineFromInput(sc, sc->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim) {
					sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_LocalInvocationID_y, logicalStoragePerThread, sc->fftDim);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				for (uint64_t k = 0; k < sc->registerBoost; ++k) {
					uint64_t t = 0;
					if (k > 0) {
						res = appendBarrierFFT(sc, 2);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadStart(sc);
						if (res != FFT_SUCCESS) return res;
						res = AppendLineFromInput(sc, sc->disableThreadsStart);
						if (res != FFT_SUCCESS) return res;
						if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim) {
							sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_LocalInvocationID_y, logicalStoragePerThread, sc->fftDim);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
					for (uint64_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
						sprintf(tempNum, "%" PRIu64 "", j * logicalGroupSize);
						res = AddReal(sc, sc->stageInvocationID, sc->gl_LocalInvocationID_y, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = MovReal(sc, sc->blockInvocationID, sc->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						sprintf(tempNum, "%" PRIu64 "", stageSize);
						res = ModReal(sc, sc->stageInvocationID, sc->stageInvocationID, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = SubReal(sc, sc->blockInvocationID, sc->blockInvocationID, sc->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						sprintf(tempNum, "%" PRIu64 "", stageRadix);
						res = MulReal(sc, sc->inoutID, sc->blockInvocationID, tempNum);
						if (res != FFT_SUCCESS) return res;
						res = AddReal(sc, sc->inoutID, sc->inoutID, sc->stageInvocationID);
						if (res != FFT_SUCCESS) return res;
						
						for (uint64_t i = 0; i < stageRadix; i++) {
							uint64_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
							id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
							sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
							t++;
							sprintf(tempNum, "%" PRIu64 "", i * stageSize);
							res = AddReal(sc, sc->sdataID, sc->inoutID, tempNum);
							if (res != FFT_SUCCESS) return res;
							res = MulReal(sc, sc->sdataID, sc->sharedStride, sc->sdataID);
							if (res != FFT_SUCCESS) return res;
							res = AddReal(sc, sc->sdataID, sc->sdataID, sc->gl_LocalInvocationID_x);
							if (res != FFT_SUCCESS) return res;
							//sprintf(sc->sdataID, "sharedStride * gl_LocalInvocationID.y + inoutID + %" PRIu64 "", i * stageSize);
							if (strcmp(stageNormalization, "")) {
								res = DivComplexNumber(sc, sc->regIDs[id], sc->regIDs[id], stageNormalization);
								if (res != FFT_SUCCESS) return res;
							}
							res = SharedStore(sc, sc->sdataID, sc->regIDs[id]);
							if (res != FFT_SUCCESS) return res;
						
						}
					}
					for (uint64_t j = logicalRegistersPerThread; j < sc->registers_per_thread; j++) {
						sprintf(tempID[t + k * sc->registers_per_thread], "%s", sc->regIDs[t + k * sc->registers_per_thread]);
						t++;
					}
					t = 0;
					if (sc->registerBoost > 1) {
						if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim)
						{
							sc->tempLen = sprintf(sc->tempStr, "	}\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(sc, sc->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadEnd(sc);
						if (res != FFT_SUCCESS) return res;
						res = appendBarrierFFT(sc, 2);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadStart(sc);
						if (res != FFT_SUCCESS) return res;
						res = AppendLineFromInput(sc, sc->disableThreadsStart);
						if (res != FFT_SUCCESS) return res;
						if (sc->localSize[1] * logicalStoragePerThreadNext > sc->fftDim) {
							sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_LocalInvocationID_y, logicalStoragePerThreadNext, sc->fftDim);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						for (uint64_t j = 0; j < logicalRegistersPerThreadNext / stageRadixNext; j++) {
							for (uint64_t i = 0; i < stageRadixNext; i++) {
								uint64_t id = j + k * logicalRegistersPerThreadNext / stageRadixNext + i * logicalRegistersPerThreadNext / stageRadixNext;
								id = (id / logicalRegistersPerThreadNext) * sc->registers_per_thread + id % logicalRegistersPerThreadNext;
								sprintf(tempNum, "%" PRIu64 "", t * logicalGroupSizeNext);
								res = AddReal(sc, sc->sdataID, sc->gl_LocalInvocationID_y, tempNum);
								if (res != FFT_SUCCESS) return res;
								res = MulReal(sc, sc->sdataID, sc->sharedStride, sc->sdataID);
								if (res != FFT_SUCCESS) return res;
								res = AddReal(sc, sc->sdataID, sc->sdataID, sc->gl_LocalInvocationID_x);
								if (res != FFT_SUCCESS) return res;
								res = SharedLoad(sc, tempID[t + k * sc->registers_per_thread], sc->sdataID);
								if (res != FFT_SUCCESS) return res;
								
								t++;
							}
						}
						if (sc->localSize[1] * logicalStoragePerThreadNext > sc->fftDim)
						{
							sc->tempLen = sprintf(sc->tempStr, "	}\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(sc, sc->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadEnd(sc);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim)
						{
							sc->tempLen = sprintf(sc->tempStr, "	}\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						res = AppendLineFromInput(sc, sc->disableThreadsEnd);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadEnd(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					sprintf(sc->regIDs[i], "%s", tempID[i]);
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
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
			tempID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
			if (tempID) {
				//resID = (char**)malloc(sizeof(char*) * sc->registers_per_thread * sc->registerBoost);
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
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
				for (uint64_t k = 0; k < sc->registerBoost; ++k) {
					for (uint64_t j = 0; j < logicalRegistersPerThread / stageRadix; j++) {
						for (uint64_t i = 0; i < stageRadix; i++) {
							uint64_t id = j + k * logicalRegistersPerThread / stageRadix + i * logicalStoragePerThread / stageRadix;
							id = (id / logicalRegistersPerThread) * sc->registers_per_thread + id % logicalRegistersPerThread;
							sprintf(tempID[j + i * logicalRegistersPerThread / stageRadix + k * sc->registers_per_thread], "%s", sc->regIDs[id]);
						}
					}
					for (uint64_t j = logicalRegistersPerThread; j < sc->registers_per_thread; j++) {
						sprintf(tempID[j + k * sc->registers_per_thread], "%s", sc->regIDs[j + k * sc->registers_per_thread]);
					}
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
					sprintf(sc->regIDs[i], "%s", tempID[i]);
				}
				for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
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
		res = appendZeropadStart(sc);
		if (res != FFT_SUCCESS) return res;
		res = AppendLineFromInput(sc, sc->disableThreadsStart);
		if (res != FFT_SUCCESS) return res;
		if (sc->localSize[1] * logicalStoragePerThread > sc->fftDim) {
			sc->tempLen = sprintf(sc->tempStr, "\
	if (%s * %" PRIu64 " < %" PRIu64 ") {\n", sc->gl_LocalInvocationID_y, logicalStoragePerThread, sc->fftDim);
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}
		if (((sc->actualInverse) && (sc->normalize)) || ((sc->convolutionStep) && (stageAngle > 0))) {
			for (uint64_t i = 0; i < logicalRegistersPerThread; i++) {
				res = DivComplexNumber(sc, sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], sc->regIDs[(i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread], stageNormalization);
				if (res != FFT_SUCCESS) return res;
			}
		}
		if (sc->localSize[1] * logicalRegistersPerThread > sc->fftDim)
		{
			sc->tempLen = sprintf(sc->tempStr, "	}\n");
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}
		res = AppendLineFromInput(sc, sc->disableThreadsEnd);
		if (res != FFT_SUCCESS) return res;
		res = appendZeropadEnd(sc);
		if (res != FFT_SUCCESS) return res;
	}
	return res;
}



FFTResult appendRadixShuffle(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t stageSize, uint64_t stageSizeSum, double stageAngle, uint64_t stageRadix, uint64_t stageRadixNext, uint64_t shuffleType) {
	FFTResult res = FFT_SUCCESS;
	switch (shuffleType) {
	case 0: case 5: case 6: case 120: case 130: case 140: case 142: {
		res = appendRadixShuffleNonStrided(sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix, stageRadixNext);
		if (res != FFT_SUCCESS) return res;
		//appendBarrierFFT(sc, 1);
		break;
	}
	case 1: case 2: case 121: case 131: case 141: case 143: {
		res = appendRadixShuffleStrided(sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, stageRadix, stageRadixNext);
		if (res != FFT_SUCCESS) return res;
		//appendBarrierFFT(sc, 1);
		break;
	}
	}
	return res;
}



FFTResult appendReorder4StepWrite(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t reorderType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char LFending[4] = "";
	if (!strcmp(floatType, "float")) sprintf(LFending, "f");

	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	char cosDef[20] = "__cosf";
	char sinDef[20] = "__sinf";
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");


	uint64_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]];// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	switch (reorderType) {
	case 1: {//grouped_c2c
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
		if ((sc->stageStartSize > 1) && (!((sc->stageStartSize > 1) && (!sc->reorderFourStep) && (sc->inverse)))) {
			if (sc->localSize[1] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim) {
				res = appendBarrierFFT(sc, 1);
				if (res != FFT_SUCCESS) return res;
				sc->writeFromRegisters = 0;
			}
			else
				sc->writeFromRegisters = 1;
			res = appendZeropadStart(sc);
			if (res != FFT_SUCCESS) return res;
			res = AppendLineFromInput(sc, sc->disableThreadsStart);
			if (res != FFT_SUCCESS) return res;
			for (uint64_t i = 0; i < sc->fftDim / sc->localSize[1]; i++) {
				uint64_t id = (i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread;
				if (sc->LUT) {
					sc->tempLen = sprintf(sc->tempStr, "		mult = twiddleLUT[%" PRIu64 "+(((%s%s)/%" PRIu64 ") %% (%" PRIu64 "))+%" PRIu64 "*(%s+%" PRIu64 ")];\n", sc->maxStageSumLUT, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (!sc->inverse) {
						sc->tempLen = sprintf(sc->tempStr, "	mult.y = -mult.y;\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "		angle = 2 * loc_PI * ((((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")) * (%s + %" PRIu64 ")) / %f%s;\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], (double)(sc->stageStartSize * sc->fftDim), LFending);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (sc->inverse) {
						if (!strcmp(floatType, "float")) {
							sc->tempLen = sprintf(sc->tempStr, "		mult.x = %s(angle);\n", cosDef);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
							sc->tempLen = sprintf(sc->tempStr, "		mult.y = %s(angle);\n", sinDef);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
							//sc->tempLen = sprintf(sc->tempStr, "		mult = %s(cos(angle), sin(angle));\n", vecType);
						}
						if (!strcmp(floatType, "double")) {
							sc->tempLen = sprintf(sc->tempStr, "		mult = sincos_20(angle);\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
					else {
						if (!strcmp(floatType, "float")) {
							sc->tempLen = sprintf(sc->tempStr, "		mult.x = %s(angle);\n", cosDef);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
							sc->tempLen = sprintf(sc->tempStr, "		mult.y = -%s(angle);\n", sinDef);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
							//sc->tempLen = sprintf(sc->tempStr, "		mult = %s(cos(angle), sin(angle));\n", vecType);
						}
						if (!strcmp(floatType, "double")) {
							sc->tempLen = sprintf(sc->tempStr, "		mult = sincos_20(-angle);\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
				}
				if (sc->writeFromRegisters) {
					sc->tempLen = sprintf(sc->tempStr, "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", sc->regIDs[id], sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", sc->regIDs[id], sc->regIDs[id], sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		%s.x = w.x;\n", sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "\
		%s = %s*(%" PRIu64 "+%s) + %s;\n", sc->inoutID, sc->sharedStride, i * sc->localSize[1], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					sc->tempLen = sprintf(sc->tempStr, "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", sc->inoutID, sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					sc->tempLen = sprintf(sc->tempStr, "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", sc->inoutID, sc->inoutID, sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		sdata[%s].x = w.x;\n", sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			res = AppendLineFromInput(sc, sc->disableThreadsEnd);
			if (res != FFT_SUCCESS) return res;
			res = appendZeropadEnd(sc);
			if (res != FFT_SUCCESS) return res;
		}
		break;
	}
	case 2: {//single_c2c_strided
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
		if (!((!sc->reorderFourStep) && (sc->inverse))) {
			if (sc->localSize[1] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim) {
				res = appendBarrierFFT(sc, 1);
				if (res != FFT_SUCCESS) return res;
				sc->writeFromRegisters = 0;
			}
			else
				sc->writeFromRegisters = 1;
			res = appendZeropadStart(sc);
			if (res != FFT_SUCCESS) return res;
			res = AppendLineFromInput(sc, sc->disableThreadsStart);
			if (res != FFT_SUCCESS) return res;
			for (uint64_t i = 0; i < sc->fftDim / sc->localSize[1]; i++) {
				uint64_t id = (i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread;
				if (sc->LUT) {
					sc->tempLen = sprintf(sc->tempStr, "		mult = twiddleLUT[%" PRIu64 " + ((%s%s) %% (%" PRIu64 ")) + (%s + %" PRIu64 ") * %" PRIu64 "];\n", sc->maxStageSumLUT, sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], sc->stageStartSize);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (!sc->inverse) {
						sc->tempLen = sprintf(sc->tempStr, "	mult.y = -mult.y;\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "		angle = 2 * loc_PI * ((((%s%s) %% (%" PRIu64 ")) * (%s + %" PRIu64 ")) / %f%s);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], (double)(sc->stageStartSize * sc->fftDim), LFending);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (sc->inverse) {
						if (!strcmp(floatType, "float")) {
							sc->tempLen = sprintf(sc->tempStr, "		mult.x = %s(angle);\n", cosDef);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
							sc->tempLen = sprintf(sc->tempStr, "		mult.y = %s(angle);\n", sinDef);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
							//sc->tempLen = sprintf(sc->tempStr, "		mult = %s(cos(angle), sin(angle));\n", vecType);
						}
						if (!strcmp(floatType, "double")) {
							sc->tempLen = sprintf(sc->tempStr, "		mult = sincos_20(angle);\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
					else {
						if (!strcmp(floatType, "float")) {
							sc->tempLen = sprintf(sc->tempStr, "		mult.x = %s(angle);\n", cosDef);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
							sc->tempLen = sprintf(sc->tempStr, "		mult.y = -%s(angle);\n", sinDef);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
							//sc->tempLen = sprintf(sc->tempStr, "		mult = %s(cos(angle), sin(angle));\n", vecType);
						}
						if (!strcmp(floatType, "double")) {
							sc->tempLen = sprintf(sc->tempStr, "		mult = sincos_20(-angle);\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
				}
				if (sc->writeFromRegisters) {
					sc->tempLen = sprintf(sc->tempStr, "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", sc->regIDs[id], sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", sc->regIDs[id], sc->regIDs[id], sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		%s.x = w.x;\n", sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "\
		%s = %s*(%" PRIu64 "+%s) + %s;\n", sc->inoutID, sc->sharedStride, i * sc->localSize[1], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					sc->tempLen = sprintf(sc->tempStr, "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", sc->inoutID, sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					sc->tempLen = sprintf(sc->tempStr, "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", sc->inoutID, sc->inoutID, sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		sdata[%s].x = w.x;\n", sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			res = AppendLineFromInput(sc, sc->disableThreadsEnd);
			if (res != FFT_SUCCESS) return res;
			res = appendZeropadEnd(sc);
			if (res != FFT_SUCCESS) return res;
		}
		break;
	}
	}
	return res;
}


FFTResult indexInputFFT(FFTSpecializationConstantsLayout* sc, const char* uintType, uint64_t inputType, const char* index_x, const char* index_y, const char* coordinate, const char* batchID) {
	FFTResult res = FFT_SUCCESS;
	switch (inputType) {
	case 0: case 2: case 3: case 4:case 5: case 6: case 120: case 130: case 140: case 142: {
		char inputOffset[30] = "";
		if (sc->inputOffset > 0)
			sprintf(inputOffset, "%" PRIu64 " + ", sc->inputOffset);
		char shiftX[500] = "";
		if (sc->inputStride[0] == 1)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, sc->inputStride[0]);
		char shiftY[500] = "";
		uint64_t mult = (sc->mergeSequencesR2C) ? 2 : 1;
		if (sc->size[1] > 1) {
			if (sc->fftDim == sc->fft_dim_full) {
				if (sc->axisSwapped) {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[0] * sc->inputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[0] * sc->inputStride[1]);
				}
				else {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[1] * sc->inputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[1] * sc->inputStride[1]);
				}
			}
			else {
				if (sc->performWorkGroupShift[1])
					sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, sc->inputStride[1]);
				else
					sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, sc->inputStride[1]);
			}
		}
		char shiftZ[500] = "";
		if (sc->size[2] > 1) {
			if (sc->performWorkGroupShift[2])
				sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->inputStride[2]);
			else
				sprintf(shiftZ, " + %s * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->inputStride[2]);
		}
		char shiftCoordinate[100] = "";
		if (sc->numCoordinates * sc->matrixConvolution > 1) {
			sprintf(shiftCoordinate, " + consts.coordinate * %" PRIu64 "", sc->inputStride[3]);
		}
		if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, sc->inputStride[3]);
		}
		char shiftBatch[100] = "";
		if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
			if (sc->convolutionStep) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, sc->inputStride[4]);
			}
			else
				sprintf(shiftBatch, " + consts.batchID * %" PRIu64 "", sc->inputStride[4]);
		}
		sc->tempLen = sprintf(sc->tempStr, "%s%s%s%s%s%s", inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 1: case 121: case 131: case 141: case 143: {
		char inputOffset[30] = "";
		if (sc->inputOffset > 0)
			sprintf(inputOffset, "%" PRIu64 " + ", sc->inputOffset);
		char shiftX[500] = "";
		if (sc->inputStride[0] == 1)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, sc->inputStride[0]);

		char shiftY[500] = "";
		if (index_y)
			sprintf(shiftY, " + (%s) * %" PRIu64 "", index_y, sc->inputStride[1]);

		char shiftZ[500] = "";
		if (sc->size[2] > 1) {
			if (sc->performWorkGroupShift[2])
				sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->inputStride[2]);
			else
				sprintf(shiftZ, " + %s * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->inputStride[2]);
		}
		char shiftCoordinate[100] = "";
		if (sc->numCoordinates * sc->matrixConvolution > 1) {
			sprintf(shiftCoordinate, " + consts.coordinate * %" PRIu64 "", sc->outputStride[3]);
		}
		if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, sc->inputStride[3]);
		}
		char shiftBatch[100] = "";
		if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
			if (sc->convolutionStep) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, sc->inputStride[4]);
			}
			else
				sprintf(shiftBatch, " + consts.batchID * %" PRIu64 "", sc->inputStride[4]);
		}
		sc->tempLen = sprintf(sc->tempStr, "%s%s%s%s%s%s", inputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	}
	return res;
}

FFTResult indexOutputFFT(FFTSpecializationConstantsLayout* sc, const char* uintType, uint64_t outputType, const char* index_x, const char* index_y, const char* coordinate, const char* batchID) {
	FFTResult res = FFT_SUCCESS;
	switch (outputType) {//single_c2c + single_c2c_strided
	case 0: case 2: case 3: case 4: case 5: case 6: case 120: case 130: case 140: case 142: {
		char outputOffset[30] = "";
		if (sc->outputOffset > 0)
			sprintf(outputOffset, "%" PRIu64 " + ", sc->outputOffset);
		char shiftX[500] = "";
		if (sc->fftDim == sc->fft_dim_full)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, sc->outputStride[0]);
		char shiftY[500] = "";
		uint64_t mult = (sc->mergeSequencesR2C) ? 2 : 1;
		if (sc->size[1] > 1) {
			if (sc->fftDim == sc->fft_dim_full) {
				if (sc->axisSwapped) {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[0] * sc->outputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[0] * sc->outputStride[1]);
				}
				else {
					if (sc->performWorkGroupShift[1])
						sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[1] * sc->outputStride[1]);
					else
						sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, mult * sc->localSize[1] * sc->outputStride[1]);
				}
			}
			else {
				if (sc->performWorkGroupShift[1])
					sprintf(shiftY, " + (%s + consts.workGroupShiftY) * %" PRIu64 "", sc->gl_WorkGroupID_y, sc->outputStride[1]);
				else
					sprintf(shiftY, " + %s * %" PRIu64 "", sc->gl_WorkGroupID_y, sc->outputStride[1]);
			}
		}
		char shiftZ[500] = "";
		if (sc->size[2] > 1) {
			if (sc->performWorkGroupShift[2])
				sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->outputStride[2]);
			else
				sprintf(shiftZ, " + %s * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->outputStride[2]);
		}
		char shiftCoordinate[100] = "";
		if (sc->numCoordinates * sc->matrixConvolution > 1) {
			sprintf(shiftCoordinate, " + consts.coordinate * %" PRIu64 "", sc->outputStride[3]);
		}
		if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, sc->outputStride[3]);
		}
		char shiftBatch[100] = "";
		if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
			if (sc->convolutionStep) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, sc->outputStride[4]);
			}
			else
				sprintf(shiftBatch, " + consts.batchID * %" PRIu64 "", sc->outputStride[4]);
		}
		sc->tempLen = sprintf(sc->tempStr, "%s%s%s%s%s%s", outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 1: case 121: case 131: case 141: case 143: {//grouped_c2c
		char outputOffset[30] = "";
		if (sc->outputOffset > 0)
			sprintf(outputOffset, "%" PRIu64 " + ", sc->outputOffset);
		char shiftX[500] = "";
		if (sc->fftDim == sc->fft_dim_full)
			sprintf(shiftX, "(%s)", index_x);
		else
			sprintf(shiftX, "(%s) * %" PRIu64 "", index_x, sc->outputStride[0]);
		char shiftY[500] = "";
		if (index_y)
			sprintf(shiftY, " + (%s) * %" PRIu64 "", index_y, sc->outputStride[1]);
		char shiftZ[500] = "";
		if (sc->size[2] > 1) {
			if (sc->performWorkGroupShift[2])
				sprintf(shiftZ, " + (%s + consts.workGroupShiftZ * %s) * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z, sc->outputStride[2]);
			else
				sprintf(shiftZ, " + %s * %" PRIu64 "", sc->gl_GlobalInvocationID_z, sc->outputStride[2]);
		}
		char shiftCoordinate[100] = "";
		if (sc->numCoordinates * sc->matrixConvolution > 1) {
			sprintf(shiftCoordinate, " + consts.coordinate * %" PRIu64 "", sc->outputStride[3]);
		}
		if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {
			sprintf(shiftCoordinate, " + %s * %" PRIu64 "", coordinate, sc->outputStride[3]);
		}
		char shiftBatch[100] = "";
		if ((sc->numBatches > 1) || (sc->numKernels > 1)) {
			if (sc->convolutionStep) {
				sprintf(shiftBatch, " + %s * %" PRIu64 "", batchID, sc->outputStride[4]);
			}
			else
				sprintf(shiftBatch, " + consts.batchID * %" PRIu64 "", sc->outputStride[4]);
		}
		sc->tempLen = sprintf(sc->tempStr, "%s%s%s%s%s%s", outputOffset, shiftX, shiftY, shiftZ, shiftCoordinate, shiftBatch);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		break;

	}
	}
	return res;
}


FFTResult appendZeropadStartReadWriteStage(FFTSpecializationConstantsLayout* sc, uint64_t readStage) {
	//return if sequence is full of zeros from the start
	FFTResult res = FFT_SUCCESS;
	if ((sc->frequencyZeropadding)) {
		switch (sc->axis_id) {
		case 0: {
			break;
		}
		case 1: {

			if (!sc->supportAxis) {
				char idX[500] = "";
				if (sc->performWorkGroupShift[0])
					sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
				else
					sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idX, sc->fft_zeropad_left_full[0], idX, sc->fft_zeropad_right_full[0]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}

			}
			break;
		}
		case 2: {
			if (!sc->supportAxis) {
				char idY[500] = "";
				if (sc->performWorkGroupShift[1])//y axis is along z workgroup here
					sprintf(idY, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
				else
					sprintf(idY, "%s", sc->gl_GlobalInvocationID_z);

				char idX[500] = "";
				if (sc->performWorkGroupShift[0])
					sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
				else
					sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
				if (sc->performZeropaddingFull[0]) {
					if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idX, sc->fft_zeropad_left_full[0], idX, sc->fft_zeropad_right_full[0]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idY, sc->fft_zeropad_left_full[1], idY, sc->fft_zeropad_right_full[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			else {
				char idY[500] = "";
				if (sc->performWorkGroupShift[1])//for support axes y is along x workgroup
					sprintf(idY, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
				else
					sprintf(idY, "%s", sc->gl_GlobalInvocationID_x);
				if (sc->performZeropaddingFull[1]) {
					if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
						sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idY, sc->fft_zeropad_left_full[1], idY, sc->fft_zeropad_right_full[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
			break;
		}
		}
	}
	else {
		switch (sc->axis_id) {
		case 0: {
			char idY[500] = "";
			char idZ[500] = "";
			uint64_t mult = (sc->mergeSequencesR2C) ? 2 : 1;

			if (readStage) {
				sprintf(idY, "(%s/%" PRIu64 ") %% %" PRIu64 "", sc->inoutID, sc->inputStride[1], sc->inputStride[2] / sc->inputStride[1]);
				sprintf(idZ, "(%s/%" PRIu64 ") %% %" PRIu64 "", sc->inoutID, sc->inputStride[2], sc->inputStride[3] / sc->inputStride[2]);

			}
			else {
				sprintf(idY, "(%s/%" PRIu64 ") %% %" PRIu64 "", sc->inoutID, sc->outputStride[1], sc->outputStride[2] / sc->outputStride[1]);
				sprintf(idZ, "(%s/%" PRIu64 ") %% %" PRIu64 "", sc->inoutID, sc->outputStride[2], sc->outputStride[3] / sc->outputStride[2]);
			}

			if (sc->performZeropaddingFull[1]) {
				if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
					sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idY, sc->fft_zeropad_left_full[1], idY, sc->fft_zeropad_right_full[1]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			if (sc->performZeropaddingFull[2]) {
				if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2]) {
					sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idZ, sc->fft_zeropad_left_full[2], idZ, sc->fft_zeropad_right_full[2]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			break;
		}
		case 1: {
			

			char idZ[500] = "";
			if (sc->performWorkGroupShift[2])
				sprintf(idZ, "(%s + consts.workGroupShiftZ * %s)", sc->gl_GlobalInvocationID_z, sc->gl_WorkGroupSize_z);
			else
				sprintf(idZ, "%s", sc->gl_GlobalInvocationID_z);
			if (sc->performZeropaddingFull[2]) {
				if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2]) {
					sc->tempLen = sprintf(sc->tempStr, "		if(!((%s >= %" PRIu64 ")&&(%s < %" PRIu64 "))) {\n", idZ, sc->fft_zeropad_left_full[2], idZ, sc->fft_zeropad_right_full[2]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}

			break;
		}
		case 2: {

			break;
		}
		}
	}
	return res;
}


FFTResult appendZeropadEndReadWriteStage(FFTSpecializationConstantsLayout* sc) {
	//return if sequence is full of zeros from the start
	FFTResult res = FFT_SUCCESS;
	if ((sc->frequencyZeropadding)) {
		switch (sc->axis_id) {
		case 0: {
			break;
		}
		case 1: {
			char idX[500] = "";
			if (sc->performWorkGroupShift[0])
				sprintf(idX, "(%s + consts.workGroupShiftX * %s)", sc->gl_GlobalInvocationID_x, sc->gl_WorkGroupSize_x);
			else
				sprintf(idX, "%s", sc->gl_GlobalInvocationID_x);
			if (sc->performZeropaddingFull[0]) {
				if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0]) {
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			break;
		}
		case 2: {
			if (sc->performZeropaddingFull[0]) {
				if (sc->fft_zeropad_left_full[0] < sc->fft_zeropad_right_full[0]) {
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			if (sc->performZeropaddingFull[1]) {
				if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			break;
		}
		}
	}
	else {
		switch (sc->axis_id) {
		case 0: {
			if (sc->performZeropaddingFull[1]) {
				if (sc->fft_zeropad_left_full[1] < sc->fft_zeropad_right_full[1]) {
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			if (sc->performZeropaddingFull[2]) {
				if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2]) {
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			break;
		}
		case 1: {
			if (sc->performZeropaddingFull[2]) {
				if (sc->fft_zeropad_left_full[2] < sc->fft_zeropad_right_full[2]) {
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			break;
		}
		case 2: {

			break;
		}
		}
	}
	return res;
}


FFTResult appendReorder4StepRead(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* uintType, uint64_t reorderType) {
	FFTResult res = FFT_SUCCESS;
	char vecType[30];
	char LFending[4] = "";
	if (!strcmp(floatType, "float")) sprintf(LFending, "f");

	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	char cosDef[20] = "__cosf";
	char sinDef[20] = "__sinf";
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");


	uint64_t logicalRegistersPerThread = sc->registers_per_thread_per_radix[sc->stageRadix[0]];// (sc->registers_per_thread % sc->stageRadix[sc->numStages - 1] == 0) ? sc->registers_per_thread : sc->min_registers_per_thread;
	switch (reorderType) {
	case 1: {//grouped_c2c
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
		if ((sc->stageStartSize > 1) && (!sc->reorderFourStep) && (sc->inverse)) {
			if (sc->localSize[1] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim) {
				res = appendBarrierFFT(sc, 1);
				if (res != FFT_SUCCESS) return res;
				sc->readToRegisters = 0;
			}
			else
				sc->readToRegisters = 1;
			res = appendZeropadStart(sc);
			if (res != FFT_SUCCESS) return res;
			res = AppendLineFromInput(sc, sc->disableThreadsStart);
			if (res != FFT_SUCCESS) return res;
			for (uint64_t i = 0; i < sc->fftDim / sc->localSize[1]; i++) {
				uint64_t id = (i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread;
				if (sc->LUT) {
					sc->tempLen = sprintf(sc->tempStr, "		mult = twiddleLUT[%" PRIu64 "+(((%s%s)/%" PRIu64 ") %% (%" PRIu64 "))+%" PRIu64 "*(%s+%" PRIu64 ")];\n", sc->maxStageSumLUT, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (!sc->inverse) {
						sc->tempLen = sprintf(sc->tempStr, "	mult.y = -mult.y;\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "		angle = 2 * loc_PI * ((((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")) * (%s + %" PRIu64 ")) / %f%s;\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], (double)(sc->stageStartSize * sc->fftDim), LFending);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (!strcmp(floatType, "float")) {
						sc->tempLen = sprintf(sc->tempStr, "		mult.x = %s(angle);\n", cosDef);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "		mult.y = %s(angle);\n", sinDef);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						//sc->tempLen = sprintf(sc->tempStr, "		mult = %s(cos(angle), sin(angle));\n", vecType);
					}
					if (!strcmp(floatType, "double")) {
						sc->tempLen = sprintf(sc->tempStr, "		mult = sincos_20(angle);\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				if (sc->readToRegisters) {
					sc->tempLen = sprintf(sc->tempStr, "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", sc->regIDs[id], sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", sc->regIDs[id], sc->regIDs[id], sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		%s.x = w.x;\n", sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "\
		%s = %s*(%" PRIu64 "+%s) + %s;\n", sc->inoutID, sc->sharedStride, i * sc->localSize[1], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					sc->tempLen = sprintf(sc->tempStr, "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", sc->inoutID, sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					sc->tempLen = sprintf(sc->tempStr, "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", sc->inoutID, sc->inoutID, sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		sdata[%s].x = w.x;\n", sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			res = AppendLineFromInput(sc, sc->disableThreadsEnd);
			if (res != FFT_SUCCESS) return res;
			res = appendZeropadEnd(sc);
			if (res != FFT_SUCCESS) return res;
		}

		break;
	}
	case 2: {//single_c2c_strided
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
		if ((!sc->reorderFourStep) && (sc->inverse)) {
			if (sc->localSize[1] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim) {
				res = appendBarrierFFT(sc, 1);
				sc->readToRegisters = 0;
			}
			else
				sc->readToRegisters = 1;
			res = appendZeropadStart(sc);
			if (res != FFT_SUCCESS) return res;
			res = AppendLineFromInput(sc, sc->disableThreadsStart);
			if (res != FFT_SUCCESS) return res;
			for (uint64_t i = 0; i < sc->fftDim / sc->localSize[1]; i++) {
				uint64_t id = (i / logicalRegistersPerThread) * sc->registers_per_thread + i % logicalRegistersPerThread;
				if (sc->LUT) {
					sc->tempLen = sprintf(sc->tempStr, "		mult = twiddleLUT[%" PRIu64 " + ((%s%s) %% (%" PRIu64 ")) + (%s + %" PRIu64 ") * %" PRIu64 "];\n", sc->maxStageSumLUT, sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], sc->stageStartSize);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (!sc->inverse) {
						sc->tempLen = sprintf(sc->tempStr, "	mult.y = -mult.y;\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "		angle = 2 * loc_PI * ((((%s%s) %% (%" PRIu64 ")) * (%s + %" PRIu64 ")) / %f%s);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->gl_LocalInvocationID_y, i * sc->localSize[1], (double)(sc->stageStartSize * sc->fftDim), LFending);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					if (!strcmp(floatType, "float")) {
						sc->tempLen = sprintf(sc->tempStr, "		mult.x = %s(angle);\n", cosDef);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "		mult.y = %s(angle);\n", sinDef);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						//sc->tempLen = sprintf(sc->tempStr, "		mult = %s(cos(angle), sin(angle));\n", vecType);
					}
					if (!strcmp(floatType, "double")) {
						sc->tempLen = sprintf(sc->tempStr, "		mult = sincos_20(angle);\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
				if (sc->readToRegisters) {
					sc->tempLen = sprintf(sc->tempStr, "\
		w.x = %s.x * mult.x - %s.y * mult.y;\n", sc->regIDs[id], sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		%s.y = %s.y * mult.x + %s.x * mult.y;\n", sc->regIDs[id], sc->regIDs[id], sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		%s.x = w.x;\n", sc->regIDs[id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "\
		%s = %s*(%" PRIu64 "+%s) + %s;\n", sc->inoutID, sc->sharedStride, i * sc->localSize[1], sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					sc->tempLen = sprintf(sc->tempStr, "\
		w.x = sdata[%s].x * mult.x - sdata[%s].y * mult.y;\n", sc->inoutID, sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					sc->tempLen = sprintf(sc->tempStr, "\
		sdata[%s].y = sdata[%s].y * mult.x + sdata[%s].x * mult.y;\n", sc->inoutID, sc->inoutID, sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, "\
		sdata[%s].x = w.x;\n", sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			res = AppendLineFromInput(sc, sc->disableThreadsEnd);
			if (res != FFT_SUCCESS) return res;
			res = appendZeropadEnd(sc);
			if (res != FFT_SUCCESS) return res;
		}
		break;
	}
	}
	return res;
};

FFTResult appendReadDataFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint64_t readType) {
	FFTResult res = FFT_SUCCESS;
	double double_PI = 3.1415926535897932384626433832795;
	char vecType[30];
	char inputsStruct[20] = "";
	char LFending[4] = "";
	if (!strcmp(floatType, "float")) sprintf(LFending, "f");

	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	if (!strcmp(floatType, "double")) sprintf(LFending, "l");
	sprintf(inputsStruct, "inputs");
	char cosDef[20] = "__cosf";
	char sinDef[20] = "__sinf";

	char convTypeLeft[20] = "";
	char convTypeRight[20] = "";
	if ((!strcmp(floatType, "float")) && (strcmp(floatTypeMemory, "float"))) {
		if ((readType == 5) || (readType == 120) || (readType == 121) || (readType == 130) || (readType == 131) || (readType == 140) || (readType == 141) || (readType == 142) || (readType == 143)) {
			sprintf(convTypeLeft, "(float)");
		}
		else {

			sprintf(convTypeLeft, "conv_float2(");
			sprintf(convTypeRight, ")");
		}
	}
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

	//appendZeropadStart(sc);
	switch (readType) {
	case 0://single_c2c
	{
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX ");
		char shiftY[500] = "";
		if (sc->axisSwapped) {
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*%s ", sc->gl_WorkGroupSize_x);
		}
		else {
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*%s ", sc->gl_WorkGroupSize_y);
		}
		char shiftY2[100] = "";
		if (sc->performWorkGroupShift[1])
			sprintf(shiftY, " + consts.workGroupShiftY ");
		if (sc->fftDim < sc->fft_dim_full) {
			if (sc->axisSwapped) {
				sc->tempLen = sprintf(sc->tempStr, "		%s numActiveThreads = ((%s/%" PRIu64 ")==%" PRIu64 ") ? %" PRIu64 " : %" PRIu64 ";\n", uintType, sc->gl_WorkGroupID_x, sc->firstStageStartSize / sc->fftDim, ((uint64_t)floor(sc->fft_dim_full / ((double)sc->localSize[0] * sc->fftDim))) / (sc->firstStageStartSize / sc->fftDim), (sc->fft_dim_full - (sc->firstStageStartSize / sc->fftDim) * ((((uint64_t)floor(sc->fft_dim_full / ((double)sc->localSize[0] * sc->fftDim))) / (sc->firstStageStartSize / sc->fftDim)) * sc->localSize[0] * sc->fftDim)) / sc->min_registers_per_thread / (sc->firstStageStartSize / sc->fftDim), sc->localSize[0] * sc->localSize[1]);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sprintf(sc->disableThreadsStart, "		if(%s * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ") < %" PRIu64 ") {\n", sc->gl_LocalInvocationID_x, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[0] * sc->firstStageStartSize, sc->fft_dim_full);
				sc->tempLen = sprintf(sc->tempStr, "		if((%s+%" PRIu64 "*%s)< numActiveThreads) {\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sprintf(sc->disableThreadsEnd, "}");
			}
			else {
				sprintf(sc->disableThreadsStart, "		if(%s * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ") < %" PRIu64 ") {\n", sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize, sc->fft_dim_full);
				res = AppendLineFromInput(sc, sc->disableThreadsStart);
				if (res != FFT_SUCCESS) return res;
				sprintf(sc->disableThreadsEnd, "}");
			}
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "		{ \n");
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}

		if ((sc->localSize[1] > 1) || ((sc->performR2C) && (sc->inverse)) || (sc->localSize[0] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim))
			sc->readToRegisters = 0;
		else
			sc->readToRegisters = 1;
		if (sc->fftDim == sc->fft_dim_full) {
			for (uint64_t k = 0; k < sc->registerBoost; k++) {
				for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {

					if (sc->localSize[1] == 1)
						sc->tempLen = sprintf(sc->tempStr, "		combinedID = %s + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
					else
						sc->tempLen = sprintf(sc->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (sc->inputStride[0] > 1)
						sc->tempLen = sprintf(sc->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") * %" PRIu64 " + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", sc->fftDim, sc->inputStride[0], sc->fftDim, sc->inputStride[1]);
					else
						sc->tempLen = sprintf(sc->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", sc->fftDim, sc->fftDim, sc->inputStride[1]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (sc->axisSwapped) {
						if (sc->size[sc->axis_id + 1] % sc->localSize[0] != 0) {
							sc->tempLen = sprintf(sc->tempStr, "		if(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 "){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY2, sc->localSize[0], sc->size[sc->axis_id + 1]);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
					else {
						if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0) {
							sc->tempLen = sprintf(sc->tempStr, "		if(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 "){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY2, sc->localSize[1], sc->size[sc->axis_id + 1]);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
					if (sc->zeropad[0]) {
						sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->inputStride[1], sc->fft_zeropad_left_read[sc->axis_id], sc->inputStride[1], sc->fft_zeropad_right_read[sc->axis_id]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					res = indexInputFFT(sc, uintType, readType, sc->inoutID, 0, requestCoordinate, requestBatch);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, ";\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					res = appendZeropadStartReadWriteStage(sc, 1);

					if (res != FFT_SUCCESS) return res;
					if (sc->readToRegisters) {
						if (sc->inputBufferBlockNum == 1)
							sc->tempLen = sprintf(sc->tempStr, "		%s = %s%s[%s]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
						else
							sc->tempLen = sprintf(sc->tempStr, "		%s = %sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
					}
					else {
						if (sc->axisSwapped) {
							if (sc->inputBufferBlockNum == 1)
								sc->tempLen = sprintf(sc->tempStr, "		sdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")] = %s%s[%s]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
							else
								sc->tempLen = sprintf(sc->tempStr, "		sdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")] = %sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
						}
						else {
							if (sc->inputBufferBlockNum == 1)
								sc->tempLen = sprintf(sc->tempStr, "		sdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride] = %s%s[%s]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
							else
								sc->tempLen = sprintf(sc->tempStr, "		sdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride] = %sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
						}
					}
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;

					res = appendZeropadEndReadWriteStage(sc);

					if (res != FFT_SUCCESS) return res;
					if (sc->zeropad[0]) {
						sc->tempLen = sprintf(sc->tempStr, "		}else{\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						if (sc->readToRegisters) {
							sc->tempLen = sprintf(sc->tempStr, "			%s.x =0;%s.y = 0;\n", sc->regIDs[i + k * sc->registers_per_thread], sc->regIDs[i + k * sc->registers_per_thread]);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (sc->axisSwapped) {
								sc->tempLen = sprintf(sc->tempStr, "			sdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")].x = 0;\n", sc->fftDim, sc->fftDim);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "			sdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")].y = 0;\n", sc->fftDim, sc->fftDim);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
							else {
								sc->tempLen = sprintf(sc->tempStr, "			sdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride].x = 0;\n", sc->fftDim, sc->fftDim);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "			sdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride].y = 0;\n", sc->fftDim, sc->fftDim);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						sc->tempLen = sprintf(sc->tempStr, "		}\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					if (sc->axisSwapped) {
						if (sc->size[sc->axis_id + 1] % sc->localSize[0] != 0) {
							sc->tempLen = sprintf(sc->tempStr, "		}");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
					else {
						if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0) {
							sc->tempLen = sprintf(sc->tempStr, "		}");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}

				}
			}

		}
		else {
			for (uint64_t k = 0; k < sc->registerBoost; k++) {
				for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {

					if (sc->axisSwapped) {
						if (sc->localSize[1] == 1)
							sc->tempLen = sprintf(sc->tempStr, "		combinedID = %s + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
						else
							sc->tempLen = sprintf(sc->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 "*numActiveThreads;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread));
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ");\n", sc->fftDim, sc->fftDim, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[0] * sc->firstStageStartSize);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						sc->tempLen = sprintf(sc->tempStr, "		inoutID = %s+%" PRIu64 "+%s * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ");\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					if (sc->zeropad[0]) {
						sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->fft_dim_full, sc->fft_zeropad_left_read[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_read[sc->axis_id]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					res = indexInputFFT(sc, uintType, readType, sc->inoutID, 0, requestCoordinate, requestBatch);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, ";\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					res = appendZeropadStartReadWriteStage(sc, 1);
					if (res != FFT_SUCCESS) return res;
					if (sc->readToRegisters) {
						if (sc->inputBufferBlockNum == 1)
							sc->tempLen = sprintf(sc->tempStr, "			%s = %s%s[%s]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
						else
							sc->tempLen = sprintf(sc->tempStr, "			%s = %sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						if (sc->axisSwapped) {

							if (sc->inputBufferBlockNum == 1)
								sc->tempLen = sprintf(sc->tempStr, "		sdata[(combinedID / %" PRIu64 ") + sharedStride*(combinedID %% %" PRIu64 ")] = %s%s[inoutID]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, inputsStruct, convTypeRight);
							else
								sc->tempLen = sprintf(sc->tempStr, "		sdata[(combinedID / %" PRIu64 ") + sharedStride*(combinedID %% %" PRIu64 ")] = %sinputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "]%s;\n", sc->fftDim, sc->fftDim, convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (sc->inputBufferBlockNum == 1)
								sc->tempLen = sprintf(sc->tempStr, "		sdata[sharedStride*%s + (%s + %" PRIu64 ")] = %s%s[inoutID]%s;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeLeft, inputsStruct, convTypeRight);
							else
								sc->tempLen = sprintf(sc->tempStr, "		sdata[sharedStride*%s + (%s + %" PRIu64 ")] = %sinputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "]%s;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeLeft, sc->inputBufferBlockSize, inputsStruct, sc->inputBufferBlockSize, convTypeRight);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
					res = appendZeropadEndReadWriteStage(sc);
					if (res != FFT_SUCCESS) return res;
					if (sc->zeropad[0]) {
						sc->tempLen = sprintf(sc->tempStr, "		}else{\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						if (sc->readToRegisters) {
							sc->tempLen = sprintf(sc->tempStr, "			%s.x = 0; %s.y = 0;\n", sc->regIDs[i + k * sc->registers_per_thread], sc->regIDs[i + k * sc->registers_per_thread]);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (sc->axisSwapped) {
								sc->tempLen = sprintf(sc->tempStr, "			sdata[(combinedID / %" PRIu64 ") + sharedStride*(combinedID %% %" PRIu64 ")].x = 0;\n", sc->fftDim, sc->fftDim);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "			sdata[(combinedID / %" PRIu64 ") + sharedStride*(combinedID %% %" PRIu64 ")].y = 0;\n", sc->fftDim, sc->fftDim);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
							else {
								sc->tempLen = sprintf(sc->tempStr, "			sdata[sharedStride*%s + (%s + %" PRIu64 ")].x = 0;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
								sc->tempLen = sprintf(sc->tempStr, "			sdata[sharedStride*%s + (%s + %" PRIu64 ")].y = 0;\n", sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						sc->tempLen = sprintf(sc->tempStr, "		}\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
		}
		sc->tempLen = sprintf(sc->tempStr, "	}\n");
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 1://grouped_c2c
	{
		if (sc->localSize[1] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim)
			sc->readToRegisters = 0;
		else
			sc->readToRegisters = 1;
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);

		sprintf(sc->disableThreadsStart, "		if (((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim * sc->stageStartSize, sc->size[sc->axis_id]);
		res = AppendLineFromInput(sc, sc->disableThreadsStart);
		if (res != FFT_SUCCESS) return res;
		sprintf(sc->disableThreadsEnd, "}");
		for (uint64_t k = 0; k < sc->registerBoost; k++) {
			for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
				sc->tempLen = sprintf(sc->tempStr, "		inoutID = (%" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 "));\n", sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim * sc->stageStartSize);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				if (sc->zeropad[0]) {
					sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->fft_dim_full, sc->fft_zeropad_left_read[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_read[sc->axis_id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				sprintf(index_x, "(%s%s) %% (%" PRIu64 ")", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x);
				res = indexInputFFT(sc, uintType, readType, index_x, sc->inoutID, requestCoordinate, requestBatch);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				res = appendZeropadStartReadWriteStage(sc, 1);
				if (res != FFT_SUCCESS) return res;
				if (sc->readToRegisters) {
					if (sc->inputBufferBlockNum == 1)
						sc->tempLen = sprintf(sc->tempStr, "			%s=%s%s[%s]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
					else
						sc->tempLen = sprintf(sc->tempStr, "			%s=%sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					if (sc->inputBufferBlockNum == 1)
						sc->tempLen = sprintf(sc->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s]=%s%s[%s]%s;\n", sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
					else
						sc->tempLen = sprintf(sc->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s]=%sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				res = appendZeropadEndReadWriteStage(sc);
				if (res != FFT_SUCCESS) return res;
				if (sc->zeropad[0]) {
					sc->tempLen = sprintf(sc->tempStr, "		}else{\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (sc->readToRegisters) {
						sc->tempLen = sprintf(sc->tempStr, "			%s.x = 0; %s.y = 0;\n", sc->regIDs[i + k * sc->registers_per_thread], sc->regIDs[i + k * sc->registers_per_thread]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						sc->tempLen = sprintf(sc->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s].x=0;\n", sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s].y=0;\n", sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
		}
		sc->tempLen = sprintf(sc->tempStr, "	}\n");
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 2://single_c2c_strided
	{
		if (sc->localSize[1] * sc->stageRadix[0] * (sc->registers_per_thread_per_radix[sc->stageRadix[0]] / sc->stageRadix[0]) > sc->fftDim)
			sc->readToRegisters = 0;
		else
			sc->readToRegisters = 1;
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);

		//sc->tempLen = sprintf(sc->tempStr, "		if(gl_GlobalInvolcationID.x%s >= %" PRIu64 ") return; \n", shiftX, sc->size[0] / axis->specializationConstants.fftDim);
		sprintf(sc->disableThreadsStart, "		if (((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim, sc->fft_dim_full);
		res = AppendLineFromInput(sc, sc->disableThreadsStart);
		if (res != FFT_SUCCESS) return res;
		sprintf(sc->disableThreadsEnd, "}");
		for (uint64_t k = 0; k < sc->registerBoost; k++) {
			for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
				sc->tempLen = sprintf(sc->tempStr, "		inoutID = (%s%s) %% (%" PRIu64 ") + %" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") * (%" PRIu64 ");\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				if (sc->zeropad[0]) {
					sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->fft_dim_full, sc->fft_zeropad_left_read[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_read[sc->axis_id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				res = indexInputFFT(sc, uintType, readType, sc->inoutID, 0, requestCoordinate, requestBatch);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				res = appendZeropadStartReadWriteStage(sc, 1);
				if (res != FFT_SUCCESS) return res;
				if (sc->readToRegisters) {
					if (sc->inputBufferBlockNum == 1)
						sc->tempLen = sprintf(sc->tempStr, "			%s=%s%s[%s]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
					else
						sc->tempLen = sprintf(sc->tempStr, "			%s=%sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", sc->regIDs[i + k * sc->registers_per_thread], convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					if (sc->inputBufferBlockNum == 1)
						sc->tempLen = sprintf(sc->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s]=%s%s[%s]%s;\n", sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, inputsStruct, sc->inoutID, convTypeRight);
					else
						sc->tempLen = sprintf(sc->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s]=%sinputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "]%s;\n", sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeLeft, sc->inoutID, sc->inputBufferBlockSize, inputsStruct, sc->inoutID, sc->inputBufferBlockSize, convTypeRight);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				res = appendZeropadEndReadWriteStage(sc);
				if (res != FFT_SUCCESS) return res;
				if (sc->zeropad[0]) {
					sc->tempLen = sprintf(sc->tempStr, "		}else{\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (sc->readToRegisters) {
						sc->tempLen = sprintf(sc->tempStr, "			%s.x = 0; %s.y = 0;\n", sc->regIDs[i + k * sc->registers_per_thread], sc->regIDs[i + k * sc->registers_per_thread]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						sc->tempLen = sprintf(sc->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s].x=0;\n", sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "			sdata[%s*(%s+%" PRIu64 ")+%s].y=0;\n", sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "		}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
		}
		sc->tempLen = sprintf(sc->tempStr, "	}\n");
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	
	
	}
	return res;
}


FFTResult appendWriteDataFFT(FFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeMemory, const char* uintType, uint64_t writeType) {
	FFTResult res = FFT_SUCCESS;
	double double_PI = 3.1415926535897932384626433832795;
	char vecType[30];
	char outputsStruct[20] = "";
	char LFending[4] = "";
	if (!strcmp(floatType, "float")) sprintf(LFending, "f");


	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
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
	if ((!strcmp(floatTypeMemory, "float")) && (strcmp(floatType, "float"))) {
		if ((writeType == 6) || (writeType == 120) || (writeType == 121) || (writeType == 130) || (writeType == 131) || (writeType == 140) || (writeType == 141) || (writeType == 142) || (writeType == 143)) {

			sprintf(convTypeLeft, "(float)");

		}
		else {

			sprintf(convTypeLeft, "conv_float2(");
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
	if (sc->convolutionStep) {
		if (sc->matrixConvolution > 1) {
			sprintf(requestCoordinate, "coordinate");
		}
	}
	char requestBatch[100] = "";
	if (sc->convolutionStep) {
		if (sc->numKernels > 1) {
			sprintf(requestBatch, "batchID");//if one buffer - multiple kernel convolution
		}
	}
	switch (writeType) {
	case 0: //single_c2c
	{
		if ((sc->localSize[1] > 1) || (sc->localSize[0] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim)) {
			sc->writeFromRegisters = 0;
			res = appendBarrierFFT(sc, 1);
			if (res != FFT_SUCCESS) return res;
		}
		else
			sc->writeFromRegisters = 1;
		//res = appendZeropadStart(sc);
		//if (res != FFT_SUCCESS) return res;
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX ");
		char shiftY[500] = "";
		if (sc->axisSwapped) {
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*%s ", sc->gl_WorkGroupSize_x);
		}
		else {
			if (sc->performWorkGroupShift[1])
				sprintf(shiftY, " + consts.workGroupShiftY*%s ", sc->gl_WorkGroupSize_y);
		}

		char shiftY2[100] = "";
		if (sc->performWorkGroupShift[1])
			sprintf(shiftY, " + consts.workGroupShiftY ");
		if (sc->fftDim < sc->fft_dim_full) {
			if (sc->axisSwapped) {
				if (!sc->reorderFourStep) {
					sc->tempLen = sprintf(sc->tempStr, "		if((%s+%" PRIu64 "*%s)< numActiveThreads) {\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					sc->tempLen = sprintf(sc->tempStr, "		if (((%s + %" PRIu64 " * %s) %% %" PRIu64 " + ((%s%s) / %" PRIu64 ")*%" PRIu64 " < %" PRIu64 ")){\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, sc->localSize[0], sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[0], sc->fft_dim_full / sc->firstStageStartSize);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
			else {
				sc->tempLen = sprintf(sc->tempStr, "		if (((%s + %" PRIu64 " * %s) %% %" PRIu64 " + ((%s%s) / %" PRIu64 ")*%" PRIu64 " < %" PRIu64 ")){\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, sc->localSize[1], sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1], sc->fft_dim_full / sc->firstStageStartSize);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
			}
		}
		else {
			sc->tempLen = sprintf(sc->tempStr, "		{ \n");
			res = AppendLine(sc);
			if (res != FFT_SUCCESS) return res;
		}


		if (sc->reorderFourStep) {

			if (sc->fftDim == sc->fft_dim_full) {
				for (uint64_t k = 0; k < sc->registerBoost; k++) {
					for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
						if (sc->localSize[1] == 1)
							sc->tempLen = sprintf(sc->tempStr, "		combinedID = %s + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
						else
							sc->tempLen = sprintf(sc->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;

						if (sc->outputStride[0] > 1)
							sc->tempLen = sprintf(sc->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") * %" PRIu64 " + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", sc->fftDim, sc->outputStride[0], sc->fftDim, sc->outputStride[1]);
						else
							sc->tempLen = sprintf(sc->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", sc->fftDim, sc->fftDim, sc->outputStride[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						if (sc->axisSwapped) {
							if (sc->size[sc->axis_id + 1] % sc->localSize[0] != 0) {
								sc->tempLen = sprintf(sc->tempStr, "		if(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 "){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY2, sc->localSize[0], sc->size[sc->axis_id + 1]);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						else {
							if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0) {
								sc->tempLen = sprintf(sc->tempStr, "		if(combinedID / %" PRIu64 " + (%s%s)*%" PRIu64 "< %" PRIu64 "){", sc->fftDim, sc->gl_WorkGroupID_y, shiftY2, sc->localSize[1], sc->size[sc->axis_id + 1]);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						if (sc->zeropad[1]) {
							sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->outputStride[1], sc->fft_zeropad_left_write[sc->axis_id], sc->outputStride[1], sc->fft_zeropad_right_write[sc->axis_id]);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;

						res = indexOutputFFT(sc, uintType, writeType, sc->inoutID, 0, requestCoordinate, requestBatch);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadStartReadWriteStage(sc, 0);
						if (res != FFT_SUCCESS) return res;
						if (sc->writeFromRegisters) {
							if (sc->outputBufferBlockNum == 1)
								sc->tempLen = sprintf(sc->tempStr, "		%s[%s] = %s%s%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							else
								sc->tempLen = sprintf(sc->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %s%s%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (sc->axisSwapped) {
								if (sc->outputBufferBlockNum == 1)
									sc->tempLen = sprintf(sc->tempStr, "		%s[%s] = %ssdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								else
									sc->tempLen = sprintf(sc->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
							else {
								if (sc->outputBufferBlockNum == 1)
									sc->tempLen = sprintf(sc->tempStr, "		%s[%s] = %ssdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								else
									sc->tempLen = sprintf(sc->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						res = appendZeropadEndReadWriteStage(sc);
						if (res != FFT_SUCCESS) return res;
						if (sc->zeropad[1]) {
							sc->tempLen = sprintf(sc->tempStr, "	}\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						if (sc->axisSwapped) {
							if (sc->size[sc->axis_id + 1] % sc->localSize[0] != 0) {
								sc->tempLen = sprintf(sc->tempStr, "		}");
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						else {
							if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0) {
								sc->tempLen = sprintf(sc->tempStr, "		}");
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
					}
				}
			}
			else {
				for (uint64_t k = 0; k < sc->registerBoost; k++) {
					for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
						if (sc->localSize[1] == 1)
							sc->tempLen = sprintf(sc->tempStr, "		combinedID = %s + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
						else
							sc->tempLen = sprintf(sc->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						if (sc->axisSwapped) {
							sc->tempLen = sprintf(sc->tempStr, "		inoutID = combinedID %% %" PRIu64 " + ((%s%s) / %" PRIu64 ")*%" PRIu64 " + ((combinedID/%" PRIu64 ") * %" PRIu64 ")+ ((%s%s) %% %" PRIu64 ") * %" PRIu64 ";\n", sc->localSize[0], sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[0], sc->localSize[0], sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (sc->localSize[1] == 1)
								sc->tempLen = sprintf(sc->tempStr, "		inoutID = (%s%s)/%" PRIu64 "+ (combinedID * %" PRIu64 ")+ ((%s%s) %% %" PRIu64 ") * %" PRIu64 ";\n", sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize);
							else
								sc->tempLen = sprintf(sc->tempStr, "		inoutID = combinedID %% %" PRIu64 " + ((%s%s) / %" PRIu64 ")*%" PRIu64 " + ((combinedID/%" PRIu64 ") * %" PRIu64 ")+ ((%s%s) %% %" PRIu64 ") * %" PRIu64 ";\n", sc->localSize[1], sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1], sc->localSize[1], sc->fft_dim_full / sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						if (sc->zeropad[1]) {
							sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						res = indexOutputFFT(sc, uintType, writeType, sc->inoutID, 0, requestCoordinate, requestBatch);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						res = appendZeropadStartReadWriteStage(sc, 0);
						if (res != FFT_SUCCESS) return res;
						if (sc->writeFromRegisters) {
							if (sc->outputBufferBlockNum == 1)
								sc->tempLen = sprintf(sc->tempStr, "			%s[%s] = %s%s%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							else
								sc->tempLen = sprintf(sc->tempStr, "			outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %s%s%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (sc->axisSwapped) {
								if (sc->outputBufferBlockNum == 1)
									sc->tempLen = sprintf(sc->tempStr, "			%s[%s] = %ssdata[(combinedID %% %s)+(combinedID/%s)*sharedStride]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_WorkGroupSize_x, convTypeRight);
								else
									sc->tempLen = sprintf(sc->tempStr, "			outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %s)+(combinedID/%s)*sharedStride]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->gl_WorkGroupSize_x, sc->gl_WorkGroupSize_x, convTypeRight);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
							else {
								if (sc->outputBufferBlockNum == 1)
									sc->tempLen = sprintf(sc->tempStr, "			%s[%s] = %ssdata[(combinedID %% %s)*sharedStride+combinedID/%s]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->gl_WorkGroupSize_y, sc->gl_WorkGroupSize_y, convTypeRight);
								else
									sc->tempLen = sprintf(sc->tempStr, "			outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %s)*sharedStride+combinedID/%s]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->gl_WorkGroupSize_y, sc->gl_WorkGroupSize_y, convTypeRight);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						res = appendZeropadEndReadWriteStage(sc);
						if (res != FFT_SUCCESS) return res;
					
						if (sc->zeropad[1]) {
							sc->tempLen = sprintf(sc->tempStr, "	}");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
				}
			}
		}

		else {
			if (sc->fftDim == sc->fft_dim_full) {
				for (uint64_t k = 0; k < sc->registerBoost; k++) {
					for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
						if (sc->localSize[1] == 1)
							sc->tempLen = sprintf(sc->tempStr, "		combinedID = %s + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
						else
							sc->tempLen = sprintf(sc->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[0] * sc->localSize[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;

						if (sc->outputStride[0] > 1)
							sc->tempLen = sprintf(sc->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") * %" PRIu64 " + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", sc->fftDim, sc->outputStride[0], sc->fftDim, sc->outputStride[1]);
						else
							sc->tempLen = sprintf(sc->tempStr, "		inoutID = (combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * %" PRIu64 ";\n", sc->fftDim, sc->fftDim, sc->outputStride[1]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						if (sc->axisSwapped) {
							if (sc->size[sc->axis_id + 1] % sc->localSize[0] != 0) {
								sc->tempLen = sprintf(sc->tempStr, "		if(combinedID / %" PRIu64 " + %s*%" PRIu64 "< %" PRIu64 "){", sc->fftDim, sc->gl_WorkGroupID_y, sc->localSize[0], sc->size[sc->axis_id + 1]);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						else {
							if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0) {
								sc->tempLen = sprintf(sc->tempStr, "		if(combinedID / %" PRIu64 " + %s*%" PRIu64 "< %" PRIu64 "){", sc->fftDim, sc->gl_WorkGroupID_y, sc->localSize[1], sc->size[sc->axis_id + 1]);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						if (sc->zeropad[1]) {
							sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->outputStride[1], sc->fft_zeropad_left_write[sc->axis_id], sc->outputStride[1], sc->fft_zeropad_right_write[sc->axis_id]);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;

						res = indexOutputFFT(sc, uintType, writeType, sc->inoutID, 0, requestCoordinate, requestBatch);

						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;

						res = appendZeropadStartReadWriteStage(sc, 0);

						if (res != FFT_SUCCESS) return res;
						if (sc->writeFromRegisters) {
							if (sc->outputBufferBlockNum == 1)
								sc->tempLen = sprintf(sc->tempStr, "		%s[%s] = %s%s%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							else
								sc->tempLen = sprintf(sc->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %s%s%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (sc->axisSwapped) {
								if (sc->outputBufferBlockNum == 1)
									sc->tempLen = sprintf(sc->tempStr, "		%s[%s] = %ssdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								else
									sc->tempLen = sprintf(sc->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %" PRIu64 ") * sharedStride + (combinedID / %" PRIu64 ")]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
							else {
								if (sc->outputBufferBlockNum == 1)
									sc->tempLen = sprintf(sc->tempStr, "		%s[%s] = %ssdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								else
									sc->tempLen = sprintf(sc->tempStr, "		outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[(combinedID %% %" PRIu64 ") + (combinedID / %" PRIu64 ") * sharedStride]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->fftDim, sc->fftDim, convTypeRight);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						res = appendZeropadEndReadWriteStage(sc);
						if (res != FFT_SUCCESS) return res;
						if (sc->zeropad[1]) {
							sc->tempLen = sprintf(sc->tempStr, "	}\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						if (sc->axisSwapped) {
							if (sc->size[sc->axis_id + 1] % sc->localSize[0] != 0) {
								sc->tempLen = sprintf(sc->tempStr, "		}");
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						else {
							if (sc->size[sc->axis_id + 1] % sc->localSize[1] != 0) {
								sc->tempLen = sprintf(sc->tempStr, "		}");
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
					}
				}
			}
			else {
				for (uint64_t k = 0; k < sc->registerBoost; k++) {
					for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
						if (sc->localSize[1] == 1)
							sc->tempLen = sprintf(sc->tempStr, "		combinedID = %s + %" PRIu64 ";\n", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0]);
						else
							sc->tempLen = sprintf(sc->tempStr, "		combinedID = (%s + %" PRIu64 " * %s) + %" PRIu64 " * numActiveThreads;\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread));
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						if (sc->axisSwapped) {
							sc->tempLen = sprintf(sc->tempStr, "		inoutID = (combinedID %% %" PRIu64 ")+(combinedID / %" PRIu64 ") * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ");", sc->fftDim, sc->fftDim, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[0] * sc->firstStageStartSize);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							sc->tempLen = sprintf(sc->tempStr, "		inoutID = %s+%" PRIu64 "+%s * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ");", sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						if (sc->zeropad[1]) {
							sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						res = indexOutputFFT(sc, uintType, writeType, sc->inoutID, 0, requestCoordinate, requestBatch);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, ";\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						//sc->tempLen = sprintf(sc->tempStr, "		inoutID = indexOutput(%s+i*%" PRIu64 "+%s * %" PRIu64 " + (((%s%s) %% %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") * %" PRIu64 ")%s%s);\n", sc->gl_LocalInvocationID_x, sc->localSize[0], sc->gl_LocalInvocationID_y, sc->firstStageStartSize, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->fftDim, sc->gl_WorkGroupID_x, shiftX, sc->firstStageStartSize / sc->fftDim, sc->localSize[1] * sc->firstStageStartSize, requestCoordinate, requestBatch);
						res = appendZeropadStartReadWriteStage(sc, 0);
						if (res != FFT_SUCCESS) return res;
						if (sc->writeFromRegisters) {
							if (sc->outputBufferBlockNum == 1)
								sc->tempLen = sprintf(sc->tempStr, "		%s[inoutID]=%s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							else
								sc->tempLen = sprintf(sc->tempStr, "		outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] = %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
						else {
							if (sc->axisSwapped) {
								if (sc->outputBufferBlockNum == 1)
									sc->tempLen = sprintf(sc->tempStr, "		%s[inoutID]=%ssdata[%s + sharedStride*(%s + %" PRIu64 ")]%s;\n", outputsStruct, convTypeLeft, sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], convTypeRight);
								else
									sc->tempLen = sprintf(sc->tempStr, "		outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] = %ssdata[%s + sharedStride*(%s + %" PRIu64 ")]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->gl_LocalInvocationID_x, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], convTypeRight);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
							else {
								if (sc->outputBufferBlockNum == 1)
									sc->tempLen = sprintf(sc->tempStr, "		%s[inoutID]=%ssdata[sharedStride*%s + (%s + %" PRIu64 ")]%s;\n", outputsStruct, convTypeLeft, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeRight);
								else
									sc->tempLen = sprintf(sc->tempStr, "		outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] = %ssdata[sharedStride*%s + (%s + %" PRIu64 ")]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->gl_LocalInvocationID_y, sc->gl_LocalInvocationID_x, (i + k * sc->min_registers_per_thread) * sc->localSize[0], convTypeRight);
								res = AppendLine(sc);
								if (res != FFT_SUCCESS) return res;
							}
						}
						appendZeropadEndReadWriteStage(sc);
						if (res != FFT_SUCCESS) return res;
						if (sc->zeropad[1]) {
							sc->tempLen = sprintf(sc->tempStr, "	}\n");
							res = AppendLine(sc);
							if (res != FFT_SUCCESS) return res;
						}
					}
				}
			}
		}

		sc->tempLen = sprintf(sc->tempStr, "	}\n");
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		break;
	}
	case 1: //grouped_c2c
	{
		if (sc->localSize[1] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim) {
			sc->writeFromRegisters = 0;
			res = appendBarrierFFT(sc, 1);
			if (res != FFT_SUCCESS) return res;
		}
		else
			sc->writeFromRegisters = 1;
		//res = appendZeropadStart(sc);
		//if (res != FFT_SUCCESS) return res;
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
		sc->tempLen = sprintf(sc->tempStr, "		if (((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->fftDim * sc->stageStartSize, sc->size[sc->axis_id]);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		if ((sc->reorderFourStep) && (sc->stageStartSize == 1)) {
			for (uint64_t k = 0; k < sc->registerBoost; k++) {
				for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
					sc->tempLen = sprintf(sc->tempStr, "		inoutID = (%s + %" PRIu64 ") * (%" PRIu64 ") + (((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")) * (%" PRIu64 ") + ((%s%s) / %" PRIu64 ");\n", sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->fft_dim_full / sc->fftDim, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->firstStageStartSize / sc->fftDim, sc->fft_dim_full / sc->firstStageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * (sc->firstStageStartSize / sc->fftDim));
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					if (sc->zeropad[1]) {
						sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sprintf(index_x, "(%s%s) %% (%" PRIu64 ")", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x);
					res = indexOutputFFT(sc, uintType, writeType, index_x, sc->inoutID, requestCoordinate, requestBatch);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, ";\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					res = appendZeropadStartReadWriteStage(sc, 0);
					if (res != FFT_SUCCESS) return res;
					if (sc->writeFromRegisters) {
						if (sc->outputBufferBlockNum == 1)
							sc->tempLen = sprintf(sc->tempStr, "			%s[%s] = %s%s%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
						else
							sc->tempLen = sprintf(sc->tempStr, "			outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %s%s%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						if (sc->outputBufferBlockNum == 1)
							sc->tempLen = sprintf(sc->tempStr, "			%s[%s] = %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", outputsStruct, sc->inoutID, convTypeLeft, sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
						else
							sc->tempLen = sprintf(sc->tempStr, "			outputBlocks[%s / %" PRIu64 "]%s[%s %% %" PRIu64 "] = %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", sc->inoutID, sc->outputBufferBlockSize, outputsStruct, sc->inoutID, sc->outputBufferBlockSize, convTypeLeft, sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;

					}
					res = appendZeropadEndReadWriteStage(sc);
					if (res != FFT_SUCCESS) return res;
					if (sc->zeropad[1]) {
						sc->tempLen = sprintf(sc->tempStr, "	}\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
		}
		else {
			for (uint64_t k = 0; k < sc->registerBoost; k++) {
				for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
					if (sc->zeropad[1]) {
						sc->tempLen = sprintf(sc->tempStr, "		inoutID = (%s + %" PRIu64 ") * %" PRIu64 " + ((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ");\n", sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->stageStartSize * sc->fftDim);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
						sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					sprintf(index_x, "(%s%s) %% (%" PRIu64 ")", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x);
					sprintf(index_y, "%" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ")", sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->stageStartSize * sc->fftDim);
					res = indexOutputFFT(sc, uintType, writeType, index_x, index_y, requestCoordinate, requestBatch);
					if (res != FFT_SUCCESS) return res;
					sc->tempLen = sprintf(sc->tempStr, ";\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
					res = appendZeropadStartReadWriteStage(sc, 0);
					if (res != FFT_SUCCESS) return res;
					//sc->tempLen = sprintf(sc->tempStr, "		inoutID = indexOutput((%s%s) %% (%" PRIu64 "), %" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") %% (%" PRIu64 ")+((%s%s) / %" PRIu64 ") * (%" PRIu64 ")%s%s);\n", sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x, sc->stageStartSize, sc->gl_GlobalInvocationID_x, shiftX, sc->fft_dim_x * sc->stageStartSize, sc->stageStartSize * sc->fftDim, requestCoordinate, requestBatch);
					if (sc->writeFromRegisters) {
						if (sc->outputBufferBlockNum == 1)
							sc->tempLen = sprintf(sc->tempStr, "			%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
						else
							sc->tempLen = sprintf(sc->tempStr, "			outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] =  %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					else {
						if (sc->outputBufferBlockNum == 1)
							sc->tempLen = sprintf(sc->tempStr, "			%s[inoutID] = %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", outputsStruct, convTypeLeft, sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
						else
							sc->tempLen = sprintf(sc->tempStr, "			outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] =  %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
					res = appendZeropadEndReadWriteStage(sc);
					if (res != FFT_SUCCESS) return res;
					if (sc->zeropad[1]) {
						sc->tempLen = sprintf(sc->tempStr, "	}\n");
						res = AppendLine(sc);
						if (res != FFT_SUCCESS) return res;
					}
				}
			}
		}
		sc->tempLen = sprintf(sc->tempStr, "	}\n");
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		break;

	}
	case 2: //single_c2c_strided
	{
		if (sc->localSize[1] * sc->stageRadix[sc->numStages - 1] * (sc->registers_per_thread_per_radix[sc->stageRadix[sc->numStages - 1]] / sc->stageRadix[sc->numStages - 1]) > sc->fftDim) {
			sc->writeFromRegisters = 0;
			res = appendBarrierFFT(sc, 1);
			if (res != FFT_SUCCESS) return res;
		}
		else
			sc->writeFromRegisters = 1;
		//res = appendZeropadStart(sc);
		//if (res != FFT_SUCCESS) return res;
		char shiftX[500] = "";
		if (sc->performWorkGroupShift[0])
			sprintf(shiftX, " + consts.workGroupShiftX * %s ", sc->gl_WorkGroupSize_x);
		sc->tempLen = sprintf(sc->tempStr, "		if (((%s%s) / %" PRIu64 ") * (%" PRIu64 ") < %" PRIu64 ") {\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim, sc->fft_dim_full);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		for (uint64_t k = 0; k < sc->registerBoost; k++) {
			for (uint64_t i = 0; i < sc->min_registers_per_thread; i++) {
				sc->tempLen = sprintf(sc->tempStr, "		inoutID = (%s%s) %% (%" PRIu64 ") + %" PRIu64 " * (%s + %" PRIu64 ") + ((%s%s) / %" PRIu64 ") * (%" PRIu64 ");\n", sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_GlobalInvocationID_x, shiftX, sc->stageStartSize, sc->stageStartSize * sc->fftDim);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				if (sc->zeropad[1]) {
					sc->tempLen = sprintf(sc->tempStr, "		if((inoutID %% %" PRIu64 " < %" PRIu64 ")||(inoutID %% %" PRIu64 " >= %" PRIu64 ")){\n", sc->fft_dim_full, sc->fft_zeropad_left_write[sc->axis_id], sc->fft_dim_full, sc->fft_zeropad_right_write[sc->axis_id]);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				sc->tempLen = sprintf(sc->tempStr, "			%s = ", sc->inoutID);
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				res = indexOutputFFT(sc, uintType, writeType, sc->inoutID, 0, requestCoordinate, requestBatch);
				if (res != FFT_SUCCESS) return res;
				sc->tempLen = sprintf(sc->tempStr, ";\n");
				res = AppendLine(sc);
				if (res != FFT_SUCCESS) return res;
				res = appendZeropadStartReadWriteStage(sc, 0);
				if (res != FFT_SUCCESS) return res;
				if (sc->writeFromRegisters) {
					if (sc->outputBufferBlockNum == 1)
						sc->tempLen = sprintf(sc->tempStr, "			%s[inoutID] = %s%s%s;\n", outputsStruct, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
					else
						sc->tempLen = sprintf(sc->tempStr, "			outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] = %s%s%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->regIDs[i + k * sc->registers_per_thread], convTypeRight);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				else {
					if (sc->outputBufferBlockNum == 1)
						sc->tempLen = sprintf(sc->tempStr, "			%s[inoutID] = %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", outputsStruct, convTypeLeft, sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
					else
						sc->tempLen = sprintf(sc->tempStr, "			outputBlocks[inoutID / %" PRIu64 "]%s[inoutID %% %" PRIu64 "] = %ssdata[%s*(%s+%" PRIu64 ") + %s]%s;\n", sc->outputBufferBlockSize, outputsStruct, sc->outputBufferBlockSize, convTypeLeft, sc->sharedStride, sc->gl_LocalInvocationID_y, (i + k * sc->min_registers_per_thread) * sc->localSize[1], sc->gl_LocalInvocationID_x, convTypeRight);
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
				res = appendZeropadEndReadWriteStage(sc);
				if (res != FFT_SUCCESS) return res;
				if (sc->zeropad[1]) {
					sc->tempLen = sprintf(sc->tempStr, "	}\n");
					res = AppendLine(sc);
					if (res != FFT_SUCCESS) return res;
				}
			}
		}
		sc->tempLen = sprintf(sc->tempStr, "	}\n");
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) return res;
		break;

	}
	}
	//res = appendZeropadEnd(sc);
	//if (res != FFT_SUCCESS) return res;
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
	uint64_t registerBoost = 1;
	for (uint64_t i = 1; i <= app->configuration.registerBoost; i++) {
		if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0)
			registerBoost = i;
	}
	if (axis_id == nonStridedAxisId) maxSingleSizeNonStrided *= registerBoost;
	uint64_t maxSequenceLengthSharedMemoryStrided = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySize / complexSize;
	uint64_t maxSingleSizeStrided = (!0) ? maxSequenceLengthSharedMemoryStrided * registerBoost : maxSequenceLengthSharedMemoryStrided;
	uint64_t numPasses = 1;
	uint64_t numPassesHalfBandwidth = 1;
	uint64_t temp;
	temp = (axis_id == nonStridedAxisId) ? (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeNonStrided) : (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)maxSingleSizeStrided);
	if (temp > 1) {//more passes than one
		for (uint64_t i = 1; i <= 1; i++) {
			if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0) {
				registerBoost = i;
			}
		}
		maxSingleSizeNonStrided = maxSequenceLengthSharedMemory * registerBoost;
		maxSingleSizeStrided = maxSequenceLengthSharedMemoryStrided * registerBoost;
		temp = (axis_id == nonStridedAxisId) ? FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeNonStrided : FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / maxSingleSizeStrided;
        numPasses = (uint64_t)ceil(log2(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id]) / log2(maxSingleSizeStrided));
		//numPasses += (uint64_t)ceil(log2(temp) / log2(maxSingleSizeStrided));
	}
	registerBoost = ((axis_id == nonStridedAxisId) && ((!app->configuration.reorderFourStep) || (numPasses == 1))) ? (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)(pow(maxSequenceLengthSharedMemoryStrided, numPasses - 1) * maxSequenceLengthSharedMemory)) : (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / (double)pow(maxSequenceLengthSharedMemoryStrided, numPasses));
	uint64_t canBoost = 0;
	for (uint64_t i = registerBoost; i <= app->configuration.registerBoost; i++) {
		if (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] % (i * i) == 0) {
			registerBoost = i;
			i = app->configuration.registerBoost + 1;
			canBoost = 1;
		}
	}
	if (((canBoost == 0) || (((FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] & (FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] - 1)) != 0))) && (registerBoost > 1)) {
		registerBoost = 1;
		numPasses++;
	}
	maxSingleSizeNonStrided = maxSequenceLengthSharedMemory * registerBoost;
	maxSingleSizeStrided = maxSequenceLengthSharedMemoryStrided * registerBoost;
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
		uint64_t registers_per_thread = 8;
		uint64_t registers_per_thread_per_radix[14] = { 0 };
		uint64_t min_registers_per_thread = 8;
		if (loc_multipliers[2] > 0) {
			if (loc_multipliers[3] > 0) {
				if (loc_multipliers[5] > 0) {
					if (loc_multipliers[7] > 0) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 15;
									break;
								case 2:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								case 3:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								default:
									registers_per_thread = 16;
									registers_per_thread_per_radix[2] = 16;
									registers_per_thread_per_radix[3] = 12;
									break;
								}
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 11;
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 15;
									break;
								case 2:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								case 3:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								default:
									registers_per_thread = 16;
									registers_per_thread_per_radix[2] = 16;
									registers_per_thread_per_radix[3] = 12;
									break;
								}
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 11;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 15;
									break;
								case 2:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								case 3:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								default:
									registers_per_thread = 16;
									registers_per_thread_per_radix[2] = 16;
									registers_per_thread_per_radix[3] = 12;
									break;
								}
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 13;
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 15;
									break;
								case 2:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								case 3:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								default:
									registers_per_thread = 16;
									registers_per_thread_per_radix[2] = 16;
									registers_per_thread_per_radix[3] = 12;
									break;
								}
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 14;
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 15;
									break;
								case 2:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								default:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								}
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 10;
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 15;
									break;
								case 2:
									registers_per_thread = 12;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								default:
									registers_per_thread = 12;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								}
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 10;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 15;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 15;
									break;
								case 2:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								default:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									break;
								}
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 10;
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 6;
									registers_per_thread_per_radix[2] = 6;
									registers_per_thread_per_radix[3] = 6;
									registers_per_thread_per_radix[5] = 5;
									min_registers_per_thread = 5;
									break;
								case 2:
									registers_per_thread = 12;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 10;
									min_registers_per_thread = 10;
									break;
								default:
									registers_per_thread = 12;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 10;
									min_registers_per_thread = 10;
									break;
								}
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;

							}
						}
					}
				}
				else
				{
					if (loc_multipliers[7] > 0) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 26;
									registers_per_thread_per_radix[2] = 22;
									registers_per_thread_per_radix[3] = 21;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 22;
									registers_per_thread_per_radix[13] = 26;
									min_registers_per_thread = 21;
									break;
								case 2:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
									break;
								default:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 22;
									registers_per_thread_per_radix[2] = 22;
									registers_per_thread_per_radix[3] = 21;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 22;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 21;
									break;
								case 2:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
									break;
								default:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
									break;
								}
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 26;
									registers_per_thread_per_radix[2] = 26;
									registers_per_thread_per_radix[3] = 21;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 26;
									min_registers_per_thread = 21;
									break;
								case 2:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 12;
									break;
								default:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 12;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 7;
									registers_per_thread_per_radix[2] = 6;
									registers_per_thread_per_radix[3] = 6;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 6;
									break;
								case 2:
									registers_per_thread = 7;
									registers_per_thread_per_radix[2] = 6;
									registers_per_thread_per_radix[3] = 6;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 6;
									break;
								default:
									registers_per_thread = 8;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 6;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 6;
									break;
								}
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 6;
									registers_per_thread_per_radix[3] = 6;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 6;
									break;
								case 2:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
									break;
								default:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 6;
									registers_per_thread_per_radix[3] = 6;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 6;
									break;
								case 2:
									registers_per_thread = 12;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
									break;
								default:
									registers_per_thread = 12;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
									break;
								}
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 6;
									registers_per_thread_per_radix[3] = 6;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 6;
									break;
								case 2:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 12;
									break;
								default:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 12;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 6;
									registers_per_thread_per_radix[2] = 6;
									registers_per_thread_per_radix[3] = 6;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 6;
									break;
								case 2:
									registers_per_thread = 12;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 12;
									break;
								default:
									registers_per_thread = 12;
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 12;
									break;
								}
							}
						}
					}
				}
			}
			else {
				if (loc_multipliers[5] > 0) {
					if (loc_multipliers[7] > 0) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
									break;
								case 2:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
									break;
								case 3:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 8;
									break;
								default:
									registers_per_thread = 16;
									registers_per_thread_per_radix[2] = 16;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
									break;
								case 2:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
									break;
								case 3:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 8;
									break;
								default:
									registers_per_thread = 16;
									registers_per_thread_per_radix[2] = 16;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
									break;
								}
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
									break;
								case 2:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
									break;
								case 3:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 8;
									break;
								default:
									registers_per_thread = 16;
									registers_per_thread_per_radix[2] = 16;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 10;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 7;
									break;
								case 2:
									registers_per_thread = 10;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 7;
									break;
								default:
									registers_per_thread = 10;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 7;
									break;
								}
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
									break;
								case 2:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
									break;
								default:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 8;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
									break;
								case 2:
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
									break;
								default:
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 8;
									break;
								}
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
									break;
								case 2:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 10;
									break;
								default:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 8;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 10;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
									break;
								case 2:
									registers_per_thread = 10;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
									break;
								default:
									registers_per_thread = 10;
									registers_per_thread_per_radix[2] = 10;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 10;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 10;
									break;
								}
							}
						}
					}
				}
				else
				{
					if (loc_multipliers[7] > 0) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
									break;
								case 2:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
									break;
								case 3:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 8;
									break;
								default:
									registers_per_thread = 16;
									registers_per_thread_per_radix[2] = 16;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
									break;
								case 2:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
									break;
								case 3:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 8;
									break;
								default:
									registers_per_thread = 16;
									registers_per_thread_per_radix[2] = 16;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
									break;
								}
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 13;
									break;
								case 2:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 13;
									break;
								case 3:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 8;
									break;
								default:
									registers_per_thread = 16;
									registers_per_thread_per_radix[2] = 16;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 13;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 14;
									break;
								case 2:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 14;
									break;
								case 3:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 14;
									break;
								default:
									registers_per_thread = 14;
									registers_per_thread_per_radix[2] = 14;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 14;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 14;
									break;
								}
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 26;
									registers_per_thread_per_radix[2] = 22;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 22;
									registers_per_thread_per_radix[13] = 26;
									min_registers_per_thread = 22;
									break;
								case 2:
									registers_per_thread = 26;
									registers_per_thread_per_radix[2] = 22;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 22;
									registers_per_thread_per_radix[13] = 26;
									min_registers_per_thread = 22;
									break;
								default:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 8;
									break;
								}
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 22;
									registers_per_thread_per_radix[2] = 22;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 22;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 22;
									break;
								case 2:
									registers_per_thread = 22;
									registers_per_thread_per_radix[2] = 22;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 22;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 22;
									break;
								case 3:
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 8;
									break;
								default:
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 8;
									break;
								}
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread = 26;
									registers_per_thread_per_radix[2] = 26;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 26;
									min_registers_per_thread = 26;
									break;
								case 2:
									registers_per_thread = 26;
									registers_per_thread_per_radix[2] = 26;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 26;
									min_registers_per_thread = 26;
									break;
								default:
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 8;
									registers_per_thread_per_radix[3] = 0;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 8;
									break;
								}
							}
							else {
								registers_per_thread = (loc_multipliers[2] > 2) ? 8 : (uint64_t)pow(2, loc_multipliers[2]);
								registers_per_thread_per_radix[2] = (loc_multipliers[2] > 2) ? 8 : (uint64_t)pow(2, loc_multipliers[2]);
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = (loc_multipliers[2] > 2) ? 8 : (uint64_t)pow(2, loc_multipliers[2]);
							}
						}
					}
				}
			}
		}
		else {
			if (loc_multipliers[3] > 0) {
				if (loc_multipliers[5] > 0) {
					if (loc_multipliers[7] > 0) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 21;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 15;
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 21;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 11;
							}
							else {
								registers_per_thread = 21;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 15;
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 21;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 11;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 21;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 15;
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 21;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 13;
							}
							else {
								registers_per_thread = 21;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 15;
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 21;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 15;
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 15;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 15;
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 11;
							}
							else {
								registers_per_thread = 15;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 15;
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 11;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 15;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 15;
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 13;
							}
							else {
								registers_per_thread = 15;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 15;
								registers_per_thread_per_radix[5] = 15;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 15;
							}
						}
					}
				}
				else
				{
					if (loc_multipliers[7] > 0) {
						if (loc_multipliers[3] == 1) {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 21;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 21;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 11;
								}
								else {
									registers_per_thread = 21;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 21;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 11;
								}
							}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 21;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 21;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 13;
								}
								else {
									registers_per_thread = 21;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 21;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 21;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 21;
								}
							}
						}
						else {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 9;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 7;
								}
								else {
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 9;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 7;
								}
							}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 9;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 7;
								}
								else {
									registers_per_thread = 9;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 9;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 7;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 7;
								}
							}
						}
					}
					else {
						if (loc_multipliers[3] == 1) {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 39;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 33;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 33;
									registers_per_thread_per_radix[13] = 39;
									min_registers_per_thread = 33;
								}
								else {
									registers_per_thread = 33;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 33;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 33;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 33;
								}
							}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 39;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 39;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 39;
									min_registers_per_thread = 39;
								}
								else {
									registers_per_thread = 3;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 3;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 3;
								}
							}
						}
						else {
							if (loc_multipliers[11] > 0) {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 9;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 9;
								}
								else {
									registers_per_thread = 11;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 9;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 11;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 9;
								}
							}
							else {
								if (loc_multipliers[13] > 0) {
									registers_per_thread = 13;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 9;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 13;
									min_registers_per_thread = 9;
								}
								else {
									registers_per_thread = 9;
									registers_per_thread_per_radix[2] = 0;
									registers_per_thread_per_radix[3] = 9;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									min_registers_per_thread = 9;
								}
							}
						}
					}
				}
			}
			else {
				if (loc_multipliers[5] > 0) {
					if (loc_multipliers[7] > 0) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 13;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 5;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 5;
							}
							else {
								registers_per_thread = 11;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 5;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 5;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 13;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 5;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 5;
							}
							else {
								registers_per_thread = 7;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 5;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 5;
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 13;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 5;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 5;
							}
							else {
								registers_per_thread = 11;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 5;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 5;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 13;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 5;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 5;
							}
							else {
								registers_per_thread = 5;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 5;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 5;
							}
						}
					}
				}
				else
				{
					if (loc_multipliers[7] > 0) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 13;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 7;
							}
							else {
								registers_per_thread = 11;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 7;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 13;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 7;
							}
							else {
								registers_per_thread = 7;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 7;
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 13;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 11;
							}
							else {
								registers_per_thread = 11;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								min_registers_per_thread = 11;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread = 13;
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								min_registers_per_thread = 13;
							}
							else {
								return FFT_ERROR_UNSUPPORTED_RADIX;
							}
						}
					}
				}
			}

		}
		registers_per_thread_per_radix[8] = registers_per_thread_per_radix[2];
		registers_per_thread_per_radix[4] = registers_per_thread_per_radix[2];
		// if ((registerBoost == 4) && (registers_per_thread % 4 != 0)) {
		// 	registers_per_thread *= 2;
		// 	for (uint64_t i = 2; i < 14; i++) {
		// 		registers_per_thread_per_radix[i] *= 2;
		// 	}
		// 	min_registers_per_thread *= 2;
		// }
		if (registers_per_thread_per_radix[8] % 8 == 0) {
			loc_multipliers[8] = loc_multipliers[2] / 3;
			loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[8] * 3;
		}
		if (registers_per_thread_per_radix[4] % 4 == 0) {
			loc_multipliers[4] = loc_multipliers[2] / 2;
			loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[4] * 2;
		}
		// if ((registerBoost == 2) && (loc_multipliers[2] == 0)) {
		// 	if (loc_multipliers[4] > 0) {
		// 		loc_multipliers[4]--;
		// 		loc_multipliers[2] = 2;
		// 	}
		// 	else {
		// 		loc_multipliers[8]--;
		// 		loc_multipliers[4]++;
		// 		loc_multipliers[2]++;
		// 	}
		// }
		// if ((registerBoost == 4) && (loc_multipliers[4] == 0)) {
		// 	loc_multipliers[8]--;
		// 	loc_multipliers[4]++;
		// 	loc_multipliers[2]++;
		// }
		uint64_t maxBatchCoalesced = ((axis_id == 0) && (((k == 0) && 0) || (numPasses == 1))) ? 1 : app->configuration.coalescedMemory / complexSize;
		
		uint64_t j = 0;
		axes[k].specializationConstants.registerBoost = registerBoost;
		axes[k].specializationConstants.registers_per_thread = registers_per_thread;
		axes[k].specializationConstants.min_registers_per_thread = min_registers_per_thread;
		for (uint64_t i = 2; i < 14; i++) {
			axes[k].specializationConstants.registers_per_thread_per_radix[i] = registers_per_thread_per_radix[i];
		}
		axes[k].specializationConstants.numStages = 0;
		axes[k].specializationConstants.fftDim = locAxisSplit[k];
		uint64_t tempRegisterBoost = registerBoost;
		uint64_t switchRegisterBoost = 0;
		if (tempRegisterBoost > 1) {
			if (loc_multipliers[tempRegisterBoost] > 0) {
				loc_multipliers[tempRegisterBoost]--;
				switchRegisterBoost = tempRegisterBoost;
			}
			else {
				for (uint64_t i = 14; i > 1; i--) {
					if (loc_multipliers[i] > 0) {
						loc_multipliers[i]--;
						switchRegisterBoost = i;
						i = 1;
					}
				}
			}
		}
		for (uint64_t i = 14; i > 1; i--) {
			if (loc_multipliers[i] > 0) {
				axes[k].specializationConstants.stageRadix[j] = i;
				loc_multipliers[i]--;
				i++;
				j++;
				axes[k].specializationConstants.numStages++;
			}
		}
		if (switchRegisterBoost > 0) {
			axes[k].specializationConstants.stageRadix[axes[k].specializationConstants.numStages] = switchRegisterBoost;
			axes[k].specializationConstants.numStages++;
		}
		else {
			if (min_registers_per_thread != registers_per_thread) {
				for (uint64_t i = 0; i < axes[k].specializationConstants.numStages; i++) {
					if (axes[k].specializationConstants.registers_per_thread_per_radix[axes[k].specializationConstants.stageRadix[i]] == min_registers_per_thread) {
						j = axes[k].specializationConstants.stageRadix[i];
						axes[k].specializationConstants.stageRadix[i] = axes[k].specializationConstants.stageRadix[0];
						axes[k].specializationConstants.stageRadix[0] = j;
						i = axes[k].specializationConstants.numStages;
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
void freeShaderGenFFT(FFTSpecializationConstantsLayout* sc) {
	if (sc->disableThreadsStart) {
		free(sc->disableThreadsStart);
		sc->disableThreadsStart = 0;
	}
	if (sc->disableThreadsStart) {
		free(sc->disableThreadsEnd);
		sc->disableThreadsEnd = 0;
	}
	if (sc->regIDs) {
		for (uint64_t i = 0; i < sc->registers_per_thread * sc->registerBoost; i++) {
			if (sc->regIDs[i]) {
				free(sc->regIDs[i]);
				sc->regIDs[i] = 0;
			}
		}
		free(sc->regIDs);
		sc->regIDs = 0;
	}
}

FFTResult shaderGenFFT(char* output, FFTSpecializationConstantsLayout* sc, const char* floatType, const char* floatTypeInputMemory, const char* floatTypeOutputMemory, const char* floatTypeKernelMemory, const char* uintType, uint64_t type) {
	FFTResult res = FFT_SUCCESS;
	//appendLicense(output);
	sc->output = output;
	sc->tempStr = (char*)malloc(sizeof(char) * sc->maxTempLength);
	if (!sc->tempStr) return FFT_ERROR_MALLOC_FAILED;
	sc->tempLen = 0;
	sc->currentLen = 0;
	char vecType[30];
	char vecTypeInput[30];
	char vecTypeOutput[30];

	if (!strcmp(floatType, "half")) sprintf(vecType, "f16vec2");
	if (!strcmp(floatType, "float")) sprintf(vecType, "float2");
	if (!strcmp(floatType, "double")) sprintf(vecType, "double2");
	if (!strcmp(floatTypeInputMemory, "half")) sprintf(vecTypeInput, "f16vec2");
	if (!strcmp(floatTypeInputMemory, "float")) sprintf(vecTypeInput, "float2");
	if (!strcmp(floatTypeInputMemory, "double")) sprintf(vecTypeInput, "double2");
	if (!strcmp(floatTypeOutputMemory, "half")) sprintf(vecTypeOutput, "f16vec2");
	if (!strcmp(floatTypeOutputMemory, "float")) sprintf(vecTypeOutput, "float2");
	if (!strcmp(floatTypeOutputMemory, "double")) sprintf(vecTypeOutput, "double2");
	sprintf(sc->gl_LocalInvocationID_x, "threadIdx.x");
	sprintf(sc->gl_LocalInvocationID_y, "threadIdx.y");
	sprintf(sc->gl_LocalInvocationID_z, "threadIdx.z");
	sprintf(sc->gl_GlobalInvocationID_x, "(threadIdx.x + blockIdx.x * blockDim.x)");
	sprintf(sc->gl_GlobalInvocationID_y, "(threadIdx.y + blockIdx.y * blockDim.y)");
	sprintf(sc->gl_GlobalInvocationID_z, "(threadIdx.z + blockIdx.z * blockDim.z)");
	sprintf(sc->gl_WorkGroupID_x, "blockIdx.x");
	sprintf(sc->gl_WorkGroupID_y, "blockIdx.y");
	sprintf(sc->gl_WorkGroupID_z, "blockIdx.z");
	sprintf(sc->gl_WorkGroupSize_x, "blockDim.x");
	sprintf(sc->gl_WorkGroupSize_y, "blockDim.y");
	sprintf(sc->gl_WorkGroupSize_z, "blockDim.z");


	sprintf(sc->stageInvocationID, "stageInvocationID");
	sprintf(sc->blockInvocationID, "blockInvocationID");
	sprintf(sc->tshuffle, "tshuffle");
	sprintf(sc->sharedStride, "sharedStride");
	sprintf(sc->combinedID, "combinedID");
	sprintf(sc->inoutID, "inoutID");
	sprintf(sc->sdataID, "sdataID");
	//sprintf(sc->tempReg, "temp");

	sc->disableThreadsStart = (char*)malloc(sizeof(char) * 500);
	if (!sc->disableThreadsStart) {
		freeShaderGenFFT(sc);
		return FFT_ERROR_MALLOC_FAILED;
	}
	sc->disableThreadsEnd = (char*)malloc(sizeof(char) * 2);
	if (!sc->disableThreadsEnd) {
		freeShaderGenFFT(sc);
		return FFT_ERROR_MALLOC_FAILED;
	}
	sc->disableThreadsStart[0] = 0;
	sc->disableThreadsEnd[0] = 0;



	// res = appendExtensions(sc, floatType, floatTypeInputMemory, floatTypeOutputMemory, floatTypeKernelMemory);
	// if (res != FFT_SUCCESS) {
	// 	freeShaderGenFFT(sc);
	// 	return res;
	// }


	res = appendConstantsFFT(sc, floatType, uintType);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}

	if ((!sc->LUT) && (!strcmp(floatType, "double"))) {
		res = appendSinCos20(sc, floatType, uintType);
		if (res != FFT_SUCCESS) {
			freeShaderGenFFT(sc);
			return res;
		}
	}

	if (strcmp(floatType, floatTypeInputMemory)) {
		res = appendConversion(sc, floatType, floatTypeInputMemory);
		if (res != FFT_SUCCESS) {
			freeShaderGenFFT(sc);
			return res;
		}
	}
	if (strcmp(floatType, floatTypeOutputMemory) && strcmp(floatTypeInputMemory, floatTypeOutputMemory)) {
		res = appendConversion(sc, floatType, floatTypeOutputMemory);
		if (res != FFT_SUCCESS) {
			freeShaderGenFFT(sc);
			return res;
		}
	}
	res = appendPushConstantsFFT(sc, floatType, uintType);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}
	uint64_t id = 0;
	res = appendInputLayoutFFT(sc, id, floatTypeInputMemory, type);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}
	id++;
	res = appendOutputLayoutFFT(sc, id, floatTypeOutputMemory, type);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}
	id++;

	if (sc->LUT) {
		res = appendLUTLayoutFFT(sc, id, floatType);
		if (res != FFT_SUCCESS) {
			freeShaderGenFFT(sc);
			return res;
		}
		id++;
	}

	uint64_t locType = (((type == 0) || (type == 5) || (type == 6) || (type == 120) || (type == 130) || (type == 140) || (type == 142)) && (sc->axisSwapped)) ? 1 : type;

	sc->tempLen = sprintf(sc->tempStr, "extern __shared__ float shared[];\n");
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}
	sc->tempLen = sprintf(sc->tempStr, "extern \"C\" __launch_bounds__(%" PRIu64 ") __global__ void FFT_main ", sc->localSize[0] * sc->localSize[1] * sc->localSize[2]);
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}


	sc->tempLen = sprintf(sc->tempStr, "(%s* inputs, %s* outputs", vecTypeInput, vecTypeOutput);
	
	
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}

	if (sc->LUT) {
		sc->tempLen = sprintf(sc->tempStr, ", %s* twiddleLUT", vecType);
		res = AppendLine(sc);
		if (res != FFT_SUCCESS) {
			freeShaderGenFFT(sc);
			return res;
		}
	}
	sc->tempLen = sprintf(sc->tempStr, ") {\n");
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}

	//sc->tempLen = sprintf(sc->tempStr, ", const PushConsts consts) {\n");
	res = appendSharedMemoryFFT(sc, floatType, uintType, locType);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}


	//if (type==0) sc->tempLen = sprintf(sc->tempStr, "return;\n");
	res = appendInitialization(sc, floatType, uintType, type);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}


	res = appendReadDataFFT(sc, floatType, floatTypeInputMemory, uintType, type);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}
	res = appendReorder4StepRead(sc, floatType, uintType, locType);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}

	res = appendBoostThreadDataReorder(sc, floatType, uintType, locType, 1);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}


	uint64_t stageSize = 1;
	uint64_t stageSizeSum = 0;
	double PI_const = 3.1415926535897932384626433832795;
	double stageAngle = (sc->inverse) ? PI_const : -PI_const;
	for (uint64_t i = 0; i < sc->numStages; i++) {
		if ((i == sc->numStages - 1) && (sc->registerBoost > 1)) {
			res = appendRadixStage(sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], locType);
			if (res != FFT_SUCCESS) {
				freeShaderGenFFT(sc);
				return res;
			}
			res = appendRegisterBoostShuffle(sc, floatType, stageSize, sc->stageRadix[i - 1], sc->stageRadix[i], stageAngle);
			if (res != FFT_SUCCESS) {
				freeShaderGenFFT(sc);
				return res;
			}
		}
		else {

			res = appendRadixStage(sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], locType);
			if (res != FFT_SUCCESS) {
				freeShaderGenFFT(sc);
				return res;
			}
			switch (sc->stageRadix[i]) {
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
			if (i == sc->numStages - 1) {
				res = appendRadixShuffle(sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], sc->stageRadix[i], locType);
				if (res != FFT_SUCCESS) {
					freeShaderGenFFT(sc);
					return res;
				}
			}
			else {
				res = appendRadixShuffle(sc, floatType, uintType, stageSize, stageSizeSum, stageAngle, sc->stageRadix[i], sc->stageRadix[i + 1], locType);
				if (res != FFT_SUCCESS) {
					freeShaderGenFFT(sc);
					return res;
				}
			}
			stageSize *= sc->stageRadix[i];
			stageAngle /= sc->stageRadix[i];
		}
	}


	res = appendBoostThreadDataReorder(sc, floatType, uintType, locType, 0);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}
	res = appendReorder4StepWrite(sc, floatType, uintType, locType);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}

	res = appendWriteDataFFT(sc, floatType, floatTypeOutputMemory, uintType, type);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}

	sc->tempLen = sprintf(sc->tempStr, "}\n");
	res = AppendLine(sc);
	if (res != FFT_SUCCESS) {
		freeShaderGenFFT(sc);
		return res;
	}
	freeShaderGenFFT(sc);


	// char fileName[30] = "../kernel/kernel_";
	// if(sc->size[2] != 1){
	// sprintf(fileName, "../kernel/kernel_%" PRIu64 "x%" PRIu64 "x%" PRIu64 ".h", sc->size[0], sc->size[1], sc->size[2]);
	// }else if(sc->size[1] != 1){
	// 	sprintf(fileName, "kernel_%" PRIu64 "x%" PRIu64 ".h", sc->size[0], sc->size[1]);
	// }else{
	// 	sprintf(fileName, "kernel_%" PRIu64 ".h", sc->size[0]);
	// }
	// FILE *fp=fopen(fileName,"a");
	// fprintf(fp, "%s",output);
	// fclose(fp);
	   FILE *fp=fopen("kernel_tmp.h","a");
   fprintf(fp, "%s",output);
   fclose(fp);
	return res;
}


FFTResult FFTPlanAxis(FFTApplication* app, FFTPlan* FFTPlan, uint64_t axis_id, uint64_t axis_upload_id, uint64_t inverse) {
	//get radix stages
	FFTResult resFFT = FFT_SUCCESS;
	hipError_t res = hipSuccess;

	FFTAxis* axis = &FFTPlan->axes[axis_id][axis_upload_id];
	axis->specializationConstants.warpSize = app->configuration.warpSize;
	axis->specializationConstants.numSharedBanks = app->configuration.numSharedBanks;
	axis->specializationConstants.useUint64 = app->configuration.useUint64;
	uint64_t complexSize;
	if (app->configuration.doublePrecision) complexSize = (2 * sizeof(double));
	
	axis->specializationConstants.complexSize = complexSize;
	axis->specializationConstants.supportAxis = 0;

	uint64_t maxSequenceLengthSharedMemory = app->configuration.sharedMemorySize / complexSize;
	uint64_t maxSequenceLengthSharedMemoryPow2 = app->configuration.sharedMemorySizePow2 / complexSize;
	uint64_t maxSingleSizeStrided = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySize / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySize / complexSize;
	uint64_t maxSingleSizeStridedPow2 = (app->configuration.coalescedMemory > complexSize) ? app->configuration.sharedMemorySizePow2 / (app->configuration.coalescedMemory) : app->configuration.sharedMemorySizePow2 / complexSize;

	axis->specializationConstants.stageStartSize = 1;
	for (uint64_t i = 0; i < axis_upload_id; i++)
		axis->specializationConstants.stageStartSize *= FFTPlan->axisSplit[axis_id][i];


	axis->specializationConstants.firstStageStartSize = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id] / FFTPlan->axisSplit[axis_id][FFTPlan->numAxisUploads[axis_id] - 1];


	if (axis_id == 0) {
		//configure radix stages
		axis->specializationConstants.fft_dim_x = axis->specializationConstants.stageStartSize;
	}
	else {
		axis->specializationConstants.fft_dim_x = FFTPlan->actualFFTSizePerAxis[axis_id][0];
	}

	if ((axis_id == 0) && ((FFTPlan->numAxisUploads[axis_id] == 1) || ((axis_upload_id == 0) && (!app->configuration.reorderFourStep)))) {
		maxSequenceLengthSharedMemory *= axis->specializationConstants.registerBoost;
		maxSequenceLengthSharedMemoryPow2 = (uint64_t)pow(2, (uint64_t)log2(maxSequenceLengthSharedMemory));
	}
	else {
		maxSingleSizeStrided *= axis->specializationConstants.registerBoost;
		maxSingleSizeStridedPow2 = (uint64_t)pow(2, (uint64_t)log2(maxSingleSizeStrided));
	}


	axis->specializationConstants.reorderFourStep = (FFTPlan->numAxisUploads[axis_id] > 1) ? app->configuration.reorderFourStep : 0;
	//uint64_t passID = FFTPlan->numAxisUploads[axis_id] - 1 - axis_upload_id;
	axis->specializationConstants.fft_dim_full = FFTPlan->actualFFTSizePerAxis[axis_id][axis_id];
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
		for (uint64_t i = 0; i < axis->specializationConstants.numStages; i++) {
			switch (axis->specializationConstants.stageRadix[i]) {
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
			dimMult *= axis->specializationConstants.stageRadix[i];
		}
		axis->specializationConstants.maxStageSumLUT = maxStageSum;
		dimMult = 1;
		if (app->configuration.doublePrecision) {
            if (axis_upload_id > 0) {
				axis->bufferLUTSize = (maxStageSum + axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim) * 2 * sizeof(double);
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

			// for (uint64_t i = 0; i < axis->specializationConstants.numStages; i++) {
			// 	std::cout<<"stageradix = "<<axis->specializationConstants.stageRadix[i]<<"    ";
			// }
			// std::cout<<"\n";

			
			for (uint64_t i = 0; i < axis->specializationConstants.numStages; i++) {
				if ((axis->specializationConstants.stageRadix[i] & (axis->specializationConstants.stageRadix[i] - 1)) == 0) {
					for (uint64_t k = 0; k < log2(axis->specializationConstants.stageRadix[i]); k++) {
						for (uint64_t j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = cos(j * double_PI / localStageSize / pow(2, k));
							tempLUT[2 * (j + localStageSum) + 1] = sin(j * double_PI / localStageSize / pow(2, k));
						}
						localStageSum += localStageSize;
					}
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
				else {
					for (uint64_t k = (axis->specializationConstants.stageRadix[i] - 1); k > 0; k--) {
						for (uint64_t j = 0; j < localStageSize; j++) {
							tempLUT[2 * (j + localStageSum)] = cos(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
							tempLUT[2 * (j + localStageSum) + 1] = sin(j * 2.0 * k / axis->specializationConstants.stageRadix[i] * double_PI / localStageSize);
						}
						localStageSum += localStageSize;
					}
					localStageSize *= axis->specializationConstants.stageRadix[i];
				}
			}				


			if (axis_upload_id > 0) {
				
				for (uint64_t i = 0; i < axis->specializationConstants.stageStartSize; i++) {
					for (uint64_t j = 0; j < axis->specializationConstants.fftDim; j++) {
						double angle = 2 * double_PI * ((i * j) / (double)(axis->specializationConstants.stageStartSize * axis->specializationConstants.fftDim));
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize)] = cos(angle);
						tempLUT[maxStageSum * 2 + 2 * (i + j * axis->specializationConstants.stageStartSize) + 1] = sin(angle);
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
				if (((axis_id == 1) || (axis_id == 2)) && (!((!app->configuration.reorderFourStep) && (FFTPlan->numAxisUploads[axis_id] > 1))) && ((axis->specializationConstants.fft_dim_full == FFTPlan->axes[0][0].specializationConstants.fft_dim_full) && (FFTPlan->numAxisUploads[axis_id] == 1) && (axis->specializationConstants.fft_dim_full < maxSingleSizeStrided / axis->specializationConstants.registerBoost))) {
					axis->bufferLUT = FFTPlan->axes[0][axis_upload_id].bufferLUT;

					axis->bufferLUTSize = FFTPlan->axes[0][axis_upload_id].bufferLUTSize;
					axis->referenceLUT = 1;
				}
				else {
					if ((axis_id == 2) && (axis->specializationConstants.fft_dim_full == FFTPlan->axes[1][0].specializationConstants.fft_dim_full)) {
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



	uint64_t* axisStride = axis->specializationConstants.inputStride;
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


	axisStride = axis->specializationConstants.outputStride;
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






	axis->specializationConstants.actualInverse = inverse;
	axis->specializationConstants.inverse = inverse;
		
	

	axis->specializationConstants.inputOffset = 0;
	axis->specializationConstants.outputOffset = 0;

	uint64_t storageComplexSize;
	if (app->configuration.doublePrecision) storageComplexSize = (2 * sizeof(double));

	uint64_t initPageSize = -1;


	
	uint64_t totalSize = 0;
	uint64_t locPageSize = initPageSize;
	if ((axis->specializationConstants.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
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

	axis->specializationConstants.inputBufferBlockSize = (uint64_t)ceil(locPageSize / (double)storageComplexSize);
	axis->specializationConstants.inputBufferBlockNum = (uint64_t)ceil(totalSize / (double)(axis->specializationConstants.inputBufferBlockSize * storageComplexSize));
			//if (axis->specializationConstants.inputBufferBlockNum == 1) axis->specializationConstants.inputBufferBlockSize = totalSize / storageComplexSize;

		
	


	totalSize = 0;
	locPageSize = initPageSize;
	if ((axis->specializationConstants.reorderFourStep == 1) && (FFTPlan->numAxisUploads[axis_id] > 1))
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
	axis->specializationConstants.outputBufferBlockSize = (uint64_t)ceil(locPageSize / (double)storageComplexSize);
	axis->specializationConstants.outputBufferBlockNum = (uint64_t)ceil(totalSize / (double)(axis->specializationConstants.outputBufferBlockSize * storageComplexSize));
		//if (axis->specializationConstants.outputBufferBlockNum == 1) axis->specializationConstants.outputBufferBlockSize = totalSize / storageComplexSize;


	if (axis->specializationConstants.inputBufferBlockNum == 0) axis->specializationConstants.inputBufferBlockNum = 1;
	if (axis->specializationConstants.outputBufferBlockNum == 0) axis->specializationConstants.outputBufferBlockNum = 1;

	axis->numBindings = 2;
	axis->specializationConstants.numBuffersBound[0] = axis->specializationConstants.inputBufferBlockNum;
	axis->specializationConstants.numBuffersBound[1] = axis->specializationConstants.outputBufferBlockNum;
	axis->specializationConstants.numBuffersBound[2] = 0;
	axis->specializationConstants.numBuffersBound[3] = 0;

	if (app->configuration.useLUT) {
		axis->specializationConstants.numBuffersBound[axis->numBindings] = 1;
		axis->numBindings++;
	}

	resFFT = FFTCheckUpdateBufferSet(app, axis, 1, 0);
	if (resFFT != FFT_SUCCESS) {
		deleteFFT(app);
		return resFFT;
	}
	resFFT = FFTUpdateBufferSet(app, FFTPlan, axis, axis_id, axis_upload_id, inverse);
	if (resFFT != FFT_SUCCESS) {
		deleteFFT(app);
		return resFFT;
	}
	{

		uint64_t maxBatchCoalesced = app->configuration.coalescedMemory / complexSize;
		axis->groupedBatch = maxBatchCoalesced;
		

		if (((FFTPlan->numAxisUploads[axis_id] == 1) && (axis_id == 0)) || ((axis_id == 0) && (!app->configuration.reorderFourStep) && (axis_upload_id == 0))) {
			axis->groupedBatch = (maxSequenceLengthSharedMemoryPow2 / axis->specializationConstants.fftDim > axis->groupedBatch) ? maxSequenceLengthSharedMemoryPow2 / axis->specializationConstants.fftDim : axis->groupedBatch;
		}
		else {
			axis->groupedBatch = (maxSingleSizeStridedPow2 / axis->specializationConstants.fftDim > 1) ? maxSingleSizeStridedPow2 / axis->specializationConstants.fftDim * axis->groupedBatch : axis->groupedBatch;
		}
		

		if ((FFTPlan->numAxisUploads[axis_id] == 2) && (axis_upload_id == 0) && (axis->specializationConstants.fftDim * maxBatchCoalesced <= maxSequenceLengthSharedMemory)) {
			axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
		}

		if ((FFTPlan->numAxisUploads[axis_id] == 3) && (axis_upload_id == 0) && (axis->specializationConstants.fftDim < maxSequenceLengthSharedMemory / (2 * complexSize))) {
			axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
		}
		if (axis->groupedBatch < maxBatchCoalesced) axis->groupedBatch = maxBatchCoalesced;
		axis->groupedBatch = (axis->groupedBatch / maxBatchCoalesced) * maxBatchCoalesced;


		if (!((axis_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1)) && !((axis_id == 0) && (axis_upload_id == 0) && (!app->configuration.reorderFourStep)) && (axis->specializationConstants.fftDim > maxSingleSizeStrided)) {
			axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
		}

		if ((app->configuration.halfThreads) && (axis->groupedBatch * axis->specializationConstants.fftDim * complexSize >= app->configuration.sharedMemorySize))
			axis->groupedBatch = (uint64_t)ceil(axis->groupedBatch / 2.0);
		if (axis->groupedBatch > app->configuration.warpSize) axis->groupedBatch = (axis->groupedBatch / app->configuration.warpSize) * app->configuration.warpSize;
		if (axis->groupedBatch > 2 * maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / (2 * maxBatchCoalesced)) * (2 * maxBatchCoalesced);
		if (axis->groupedBatch > 4 * maxBatchCoalesced) axis->groupedBatch = (axis->groupedBatch / (4 * maxBatchCoalesced)) * (2 * maxBatchCoalesced);
		uint64_t maxThreadNum = maxSequenceLengthSharedMemory / (axis->specializationConstants.min_registers_per_thread * axis->specializationConstants.registerBoost);
		axis->specializationConstants.axisSwapped = 0;
		uint64_t r2cmult = (axis->specializationConstants.mergeSequencesR2C) ? 2 : 1;

		if (axis_id == 0) {
			if (axis_upload_id == 0) {

				axis->axisBlock[0] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost : 1;


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
					if (((FFTPlan->numAxisUploads[0] > 1) && (((FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim) % i) == 0)) || ((FFTPlan->numAxisUploads[0] == 1) && (((FFTPlan->actualFFTSizePerAxis[axis_id][1] / r2cmult) % i) == 0))) {
						if (i * axis->specializationConstants.fftDim * complexSize <= app->configuration.sharedMemorySize) axis->axisBlock[1] = i;
						i = 2 * currentAxisBlock1;
					}
				}

				if ((FFTPlan->numAxisUploads[0] > 1) && ((uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim) < axis->axisBlock[1])) axis->axisBlock[1] = (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim);
				if ((axis->specializationConstants.mergeSequencesR2C != 0) && (axis->specializationConstants.fftDim * axis->axisBlock[1] >= maxSequenceLengthSharedMemory)) {
					axis->specializationConstants.mergeSequencesR2C = 0;
					r2cmult = 1;
				}
				if ((FFTPlan->numAxisUploads[0] == 1) && ((uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)r2cmult) < axis->axisBlock[1])) axis->axisBlock[1] = (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)r2cmult);

				if (axis->axisBlock[1] > app->configuration.maxComputeWorkGroupSize[1]) axis->axisBlock[1] = app->configuration.maxComputeWorkGroupSize[1];
				if (axis->axisBlock[0] * axis->axisBlock[1] > app->configuration.maxThreadsNum) axis->axisBlock[1] /= 2;
				while ((axis->axisBlock[1] * (axis->specializationConstants.fftDim / axis->specializationConstants.registerBoost)) > maxSequenceLengthSharedMemory) axis->axisBlock[1] /= 2;
				if (((axis->specializationConstants.fftDim % 2 == 0) || (axis->axisBlock[0] < app->configuration.numSharedBanks / 4)) && (!((!app->configuration.reorderFourStep) && (FFTPlan->numAxisUploads[0] > 1))) && (axis->axisBlock[1] > 1) && (axis->axisBlock[1] * axis->specializationConstants.fftDim < maxSequenceLengthSharedMemoryPow2)) {


					uint64_t temp = axis->axisBlock[1];
					axis->axisBlock[1] = axis->axisBlock[0];
					axis->axisBlock[0] = temp;
					axis->specializationConstants.axisSwapped = 1;
				}
				axis->axisBlock[2] = 1;
				axis->axisBlock[3] = axis->specializationConstants.fftDim;

			}
			else {

				axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost : 1;
				uint64_t scale = app->configuration.aimThreads / axis->axisBlock[1] / axis->groupedBatch;
				if (scale > 1) axis->groupedBatch *= scale;
				axis->axisBlock[0] = (axis->specializationConstants.stageStartSize > axis->groupedBatch) ? axis->groupedBatch : axis->specializationConstants.stageStartSize;
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
				axis->axisBlock[3] = axis->specializationConstants.fftDim;
			}

		}
		if (axis_id == 1) {

			axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost : 1;

		
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
			axis->axisBlock[3] = axis->specializationConstants.fftDim;

		}
		if (axis_id == 2) {
			axis->axisBlock[1] = (axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost > 1) ? axis->specializationConstants.fftDim / axis->specializationConstants.min_registers_per_thread / axis->specializationConstants.registerBoost : 1;


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
			axis->axisBlock[3] = axis->specializationConstants.fftDim;
		}



		uint64_t tempSize[3] = { FFTPlan->actualFFTSizePerAxis[axis_id][0], FFTPlan->actualFFTSizePerAxis[axis_id][1], FFTPlan->actualFFTSizePerAxis[axis_id][2] };


		if (axis_id == 0) {
			if (axis_upload_id == 0)
				tempSize[0] = FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim / axis->axisBlock[1];
			else
				tempSize[0] = FFTPlan->actualFFTSizePerAxis[axis_id][0] / axis->specializationConstants.fftDim / axis->axisBlock[0];
			
			if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
			else  axis->specializationConstants.performWorkGroupShift[0] = 0;
			if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
			else  axis->specializationConstants.performWorkGroupShift[1] = 0;
			if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
			else  axis->specializationConstants.performWorkGroupShift[2] = 0;
		}
		if (axis_id == 1) {
			tempSize[0] = (0) ? (uint64_t)ceil((FFTPlan->actualFFTSizePerAxis[axis_id][0] / 2 + 1) / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)axis->specializationConstants.fftDim) : (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][1] / (double)axis->specializationConstants.fftDim);
			tempSize[1] = 1;
			tempSize[2] = FFTPlan->actualFFTSizePerAxis[axis_id][2];
			//if (app->configuration.actualPerformR2C == 1) tempSize[0] = (uint64_t)ceil(tempSize[0] / 2.0);
			//if (app->configuration.performZeropadding[2]) tempSize[2] = (uint64_t)ceil(tempSize[2] / 2.0);

			if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
			else  axis->specializationConstants.performWorkGroupShift[0] = 0;
			if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
			else  axis->specializationConstants.performWorkGroupShift[1] = 0;
			if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
			else  axis->specializationConstants.performWorkGroupShift[2] = 0;

		}
		if (axis_id == 2) {
			tempSize[0] = (0) ? (uint64_t)ceil((FFTPlan->actualFFTSizePerAxis[axis_id][0] / 2 + 1) / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][2] / (double)axis->specializationConstants.fftDim) : (uint64_t)ceil(FFTPlan->actualFFTSizePerAxis[axis_id][0] / (double)axis->axisBlock[0] * FFTPlan->actualFFTSizePerAxis[axis_id][2] / (double)axis->specializationConstants.fftDim);
			tempSize[1] = 1;
			tempSize[2] = FFTPlan->actualFFTSizePerAxis[axis_id][1];
			//if (app->configuration.actualPerformR2C == 1) tempSize[0] = (uint64_t)ceil(tempSize[0] / 2.0);

			if (tempSize[0] > app->configuration.maxComputeWorkGroupCount[0]) axis->specializationConstants.performWorkGroupShift[0] = 1;
			else  axis->specializationConstants.performWorkGroupShift[0] = 0;
			if (tempSize[1] > app->configuration.maxComputeWorkGroupCount[1]) axis->specializationConstants.performWorkGroupShift[1] = 1;
			else  axis->specializationConstants.performWorkGroupShift[1] = 0;
			if (tempSize[2] > app->configuration.maxComputeWorkGroupCount[2]) axis->specializationConstants.performWorkGroupShift[2] = 1;
			else  axis->specializationConstants.performWorkGroupShift[2] = 0;

		}
		
		axis->specializationConstants.localSize[0] = axis->axisBlock[0];
		axis->specializationConstants.localSize[1] = axis->axisBlock[1];
		axis->specializationConstants.localSize[2] = axis->axisBlock[2];

		


		axis->specializationConstants.numBatches = 1;
		axis->specializationConstants.numKernels = 1;
		axis->specializationConstants.sharedMemSize = app->configuration.sharedMemorySize;
		axis->specializationConstants.sharedMemSizePow2 = app->configuration.sharedMemorySizePow2;
		axis->specializationConstants.normalize = app->configuration.normalize;
		axis->specializationConstants.size[0] = FFTPlan->actualFFTSizePerAxis[axis_id][0];
		axis->specializationConstants.size[1] = FFTPlan->actualFFTSizePerAxis[axis_id][1];
		axis->specializationConstants.size[2] = FFTPlan->actualFFTSizePerAxis[axis_id][2];
		axis->specializationConstants.axis_id = axis_id;
		axis->specializationConstants.axis_upload_id = axis_upload_id;

		axis->specializationConstants.zeropad[0] = 0;
		axis->specializationConstants.zeropad[1] = 0;


    
		axis->specializationConstants.convolutionStep = 0;

		char floatTypeInputMemory[10];
		char floatTypeOutputMemory[10];
		char floatTypeKernelMemory[10];
		char floatType[10];
		axis->specializationConstants.unroll = 1;
		axis->specializationConstants.LUT = app->configuration.useLUT;
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
		//if ((axis->specializationConstants.fftDim == 8 * maxSequenceLengthSharedMemory) && (app->configuration.registerBoost >= 8)) axis->specializationConstants.registerBoost = 8;
		// if ((axis_id == 0) && (!axis->specializationConstants.inverse) && (FFTPlan->actualPerformR2CPerAxis[axis_id])) type = 5;
		// if ((axis_id == 0) && (axis->specializationConstants.inverse) && (FFTPlan->actualPerformR2CPerAxis[axis_id])) type = 6;
		


		axis->specializationConstants.cacheShuffle = 0;


		axis->specializationConstants.maxCodeLength = app->configuration.maxCodeLength;
		axis->specializationConstants.maxTempLength = app->configuration.maxTempLength;
		char* code0 = (char*)malloc(sizeof(char) * app->configuration.maxCodeLength);
		if (!code0) {
			deleteFFT(app);
			return FFT_ERROR_MALLOC_FAILED;
		}
		shaderGenFFT(code0, &axis->specializationConstants, floatType, floatTypeInputMemory, floatTypeOutputMemory, floatTypeKernelMemory, uintType, type);

		hiprtcProgram prog;
		/*char* includeNames = (char*)malloc(sizeof(char)*100);
		char* headers = (char*)malloc(sizeof(char) * 100);
		sprintf(headers, "C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.1//include//cuComplex.h");
		sprintf(includeNames, "cuComplex.h");*/
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
		if (axis->specializationConstants.usedSharedMemory > app->configuration.sharedMemorySizeStatic) {
			result2 = hipFuncSetAttribute(axis->FFTKernel, hipFuncAttributeMaxDynamicSharedMemorySize, (int)axis->specializationConstants.usedSharedMemory);
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
		size_t size = (app->configuration.useUint64) ? sizeof(FFTPushConstantsLayoutUint64) : sizeof(FFTPushConstantsLayoutUint32);
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
	if (axis->specializationConstants.axisSwapped) {//swap back for correct dispatch
		uint64_t temp = axis->axisBlock[1];
		axis->axisBlock[1] = axis->axisBlock[0];
		axis->axisBlock[0] = temp;
		axis->specializationConstants.axisSwapped = 0;
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
	app->configuration.registerBoost = 1;

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
	app->configuration.registerBoost = 1;

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

	if (axis->specializationConstants.LUT) {
		launchArgs->args[2] = &axis->bufferLUT;
	}
	for(int i = 0; i < 3; i++){
		launchArgs->gridSize[i] = (unsigned int)dispatchBlock[i];
		launchArgs->blockSize[i] = (unsigned int)axis->specializationConstants.localSize[i];
	}
	launchArgs->sharedMem = (unsigned int)axis->specializationConstants.usedSharedMemory;
		
	//printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",maxBlockSize[0], maxBlockSize[1], maxBlockSize[2], axis->specializationConstants.localSize[0], axis->specializationConstants.localSize[1], axis->specializationConstants.localSize[2]);
				
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

	resFFT = FFTCheckUpdateBufferSet(app, 0, 0, launchParams);
	if (resFFT != FFT_SUCCESS) {
		return resFFT;
	}


	if (inverse != 1) {
		for (int64_t l = (int64_t)app->localFFTPlan->numAxisUploads[0] - 1; l >= 0; l--) {
			FFTAxis* axis = &app->localFFTPlan->axes[0][l];
			FFTLaunchArgs* launchArgs = &app->localFFTPlan->launchArgs[0][l];
			resFFT = FFTUpdateBufferSet(app, app->localFFTPlan, axis, 0, l, 0);
			if (resFFT != FFT_SUCCESS) return resFFT;
			
			uint64_t dispatchBlock[3];
			if (l == 0) {
				if (app->localFFTPlan->numAxisUploads[0] > 2) {
					dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]) / (double)app->localFFTPlan->axisSplit[0][1]) * app->localFFTPlan->axisSplit[0][1];
					dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
				}
				else {
					if (app->localFFTPlan->numAxisUploads[0] > 1) {
						dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]));
						dispatchBlock[1] = app->localFFTPlan->actualFFTSizePerAxis[0][1];
					}
					else {
						dispatchBlock[0] = app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim;
						dispatchBlock[1] = (uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);

					}
				}
			}
			else {
				dispatchBlock[0] = (uint64_t)ceil(app->localFFTPlan->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
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
			    resFFT = FFTUpdateBufferSet(app, app->localFFTPlan, axis, 1, l, 0);
			    if (resFFT != FFT_SUCCESS) return resFFT;
			    uint64_t dispatchBlock[3];
				dispatchBlock[0] = (uint64_t)ceil(localSize0[1] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[1][1] / (double)axis->specializationConstants.fftDim);
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
				resFFT = FFTUpdateBufferSet(app, app->localFFTPlan, axis, 2, l, 0);
				if (resFFT != FFT_SUCCESS) return resFFT;
				uint64_t dispatchBlock[3];
				dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan->actualFFTSizePerAxis[2][2] / (double)axis->specializationConstants.fftDim);
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
				resFFT = FFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 2, l, 1);
				if (resFFT != FFT_SUCCESS) return resFFT;
				uint64_t dispatchBlock[3];
				dispatchBlock[0] = (uint64_t)ceil(localSize0[2] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[2][2] / (double)axis->specializationConstants.fftDim);
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
				resFFT = FFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 1, l, 1);
				if (resFFT != FFT_SUCCESS) return resFFT;
				uint64_t dispatchBlock[3];
				dispatchBlock[0] = (uint64_t)ceil(localSize0[1] / (double)axis->axisBlock[0] * app->localFFTPlan_inverse->actualFFTSizePerAxis[1][1] / (double)axis->specializationConstants.fftDim);
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
			resFFT = FFTUpdateBufferSet(app, app->localFFTPlan_inverse, axis, 0, l, 1);
			if (resFFT != FFT_SUCCESS) return resFFT;
			uint64_t dispatchBlock[3];
			if (l == 0) {
				if (app->localFFTPlan_inverse->numAxisUploads[0] > 2) {
					dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]) / (double)app->localFFTPlan_inverse->axisSplit[0][1]) * app->localFFTPlan_inverse->axisSplit[0][1];
					dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
				}
				else {
					if (app->localFFTPlan_inverse->numAxisUploads[0] > 1) {
						dispatchBlock[0] = (uint64_t)ceil((uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[1]));
						dispatchBlock[1] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1];
					}
					else {
						dispatchBlock[0] = app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim;
						dispatchBlock[1] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][1] / (double)axis->axisBlock[1]);
					}
				}
			}
			else {
				dispatchBlock[0] = (uint64_t)ceil(app->localFFTPlan_inverse->actualFFTSizePerAxis[0][0] / axis->specializationConstants.fftDim / (double)axis->axisBlock[0]);
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











