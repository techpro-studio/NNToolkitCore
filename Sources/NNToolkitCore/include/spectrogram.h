//
//  spectrogram.h
//  mac_test
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef spectrogram_h
#define spectrogram_h

#include <stdio.h>
#include "stdbool.h"
#include <Accelerate/Accelerate.h>

#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    int nfft;
    int noverlap;
    int step;
    int inputSize;
    int nfreq;
    int ntimeSeries;
    float fftNormalizationFactor;
    bool complex;
} SpectrogramConfig;

SpectrogramConfig SpectrogramConfigCreate(int nfft, int noverlap, int inputSize, bool complex, float fftNormalizationFactor);


typedef struct {
    SpectrogramConfig config;
    void* fftSetup;
    float *window;
} SpectrogramFilter;

typedef enum {
    FFTWindowTypeHamming = 1,
    FFTWindowTypeHann = 2,
    FFTWindowTypeBlackMan = 3,
} FFTWindowType;

SpectrogramFilter* SpectrogramFilterCreate(SpectrogramConfig config);

void SpectrogramFilterSetWindowType(SpectrogramFilter *filter, FFTWindowType type);

void SpectrogramFilterApply(SpectrogramFilter *filter, const float *input, float* output);

void SpectrogramFilterDestroy(SpectrogramFilter *filter);

#if defined __cplusplus
}
#endif


#endif /* spectrogram_h */
