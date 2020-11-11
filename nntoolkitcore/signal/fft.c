//
// Created by Alex on 10.11.2020.
//

#include "nntoolkitcore/signal/fft.h"
#include "stdlib.h"


#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #define NO
#endif


struct DFTSetupStruct{
    DFTConfig config;
    void *implementer;
};

DFTSetup DFTSetupCreate(DFTConfig config) {
    DFTSetup setup = malloc(sizeof(struct DFTSetupStruct));
    setup->config = config;
#ifdef __APPLE__
    setup->implementer = vDSP_DFT_zop_CreateSetup(NULL, config.nfft, config.forward ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE);
#else

#endif
    return setup;
}

void DFTPerform(DFTSetup setup, complex_float_spl* input, complex_float_spl* output){
#ifdef __APPLE__
    vDSP_DFT_Execute(setup->implementer,
                     input->real_p, input->imag_p,
                     output->real_p, output->imag_p);
#else

#endif
}

void DFTSetupDestroy(DFTSetup setup) {
#ifdef __APPLE__
    vDSP_DFT_DestroySetup(setup->implementer);
    free(setup);
#else

#endif
}

DFTConfig DFTConfigCreate(int nfft, bool forward, bool complex) {
    DFTConfig result;
    result.nfft = nfft;
    result.complex = complex;
    result.forward = forward;
    return result;
}
