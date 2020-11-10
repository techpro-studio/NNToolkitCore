//
// Created by Alex on 10.11.2020.
//

#include "fft.h"

#define APPLE 1

#if APPLE
    #include <Accelerate/Accelerate.h>
#else
    #define NO
#endif


struct DFTSetupStruct{
    FFTConfig config;
    void *implementer;
};

DFTSetup DFTSetupCreate(FFTConfig config) {
    DFTSetup setup = malloc(sizeof(struct DFTSetupStruct));
    setup->config = config;
#if APPLE
    setup->implementer = vDSP_DFT_zop_CreateSetup(NULL, config.nfft, config.forward ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE);
#else

#endif
    return setup;
}

void DFTPerform(DFTSetup setup, float *input, float *output) {
    
}

void DFTSetupDestroy(DFTSetup setup) {

}
