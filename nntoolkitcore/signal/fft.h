//
// Created by Alex on 10.11.2020.
//

#ifndef fft_h
#define fft_h

#include "stdbool.h"
#include "nntoolkitcore/core/loop.h"
#include "nntoolkitcore/core/types.h"

typedef struct DFTSetupStruct* DFTSetup;

typedef struct {
    int nfft;
    bool forward;
    bool complex;
} DFTConfig;

typedef struct {
    float real;
    float imag;
} ComplexFloat;

typedef struct {
    float *real_p;
    float *imag_p;
} ComplexFloatSplit;

DFTConfig DFTConfigCreate(int nfft, bool forward, bool complex);

DFTSetup DFTSetupCreate(DFTConfig config);

void DFTPerform(DFTSetup setup, ComplexFloatSplit* input, ComplexFloatSplit* output);

void DFTSetupDestroy(DFTSetup setup);

void split_complex(ComplexFloatSplit*split, ComplexFloat *complex, int size);

void join_complex_split(ComplexFloat *complex, ComplexFloatSplit*split, int size);

#endif //fft_h
