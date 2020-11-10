//
// Created by Alex on 10.11.2020.
//

#ifndef fft_h
#define fft_h

#include "stdbool.h"

typedef struct DFTSetupStruct* DFTSetup;

typedef struct {
    int nfft;
    bool forward;
    bool complex;
} FFTConfig;

typedef struct {
    float *real_p;
    float *imag_p;
} SplitComplex;

DFTSetup FFTSetupCreate(FFTConfig config);

void FFTPerform(DFTSetup setup, float *input, float *output);

void FFTSetupDestroy(DFTSetup setup);


#endif //fft_h
