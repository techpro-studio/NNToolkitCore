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
#include "window.h"

#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    int nfft;
    int noverlap;
    int step;
    int input_size;
    int nfreq;
    int ntime_series;
    float fft_normalization_factor;
    bool complex;
} SpectrogramConfig;

SpectrogramConfig SpectrogramConfigCreate(int nfft, int noverlap, int input_size, bool complex, float fft_normalization_factor);

typedef struct SpectrogramStruct* Spectrogram;

Spectrogram SpectrogramCreate(SpectrogramConfig config);

void SpectrogramSetWindowFunc(Spectrogram filter, window_fn fn);

void SpectrogramApply(Spectrogram filter, const float *input, float* output);

void SpectrogramDestroy(Spectrogram filter);

#if defined __cplusplus
}
#endif


#endif /* spectrogram_h */
