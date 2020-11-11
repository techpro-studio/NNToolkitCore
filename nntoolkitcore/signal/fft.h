//
// Created by Alex on 10.11.2020.
//

#ifndef fft_h
#define fft_h

#include "stdbool.h"
#include "nn_toolkit_core/core/loop.h"
#include "nn_toolkit_core/core/types.h"

typedef struct DFTSetupStruct* DFTSetup;

typedef struct {
    int nfft;
    bool forward;
    bool complex;
} DFTConfig;


DFTConfig DFTConfigCreate(int nfft, bool forward, bool complex);

DFTSetup DFTSetupCreate(DFTConfig config);

void DFTPerform(DFTSetup setup, complex_float_spl* input, complex_float_spl* output);

void DFTSetupDestroy(DFTSetup setup);


#endif //fft_h
