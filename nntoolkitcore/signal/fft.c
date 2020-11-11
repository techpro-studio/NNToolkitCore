//
// Created by Alex on 10.11.2020.
//

#include "nntoolkitcore/signal/fft.h"
#include "stdlib.h"

#if APPLE
    #include <Accelerate/Accelerate.h>
#else

#endif


struct DFTSetupStruct{
    DFTConfig config;
    void *implementer;
};

DFTSetup DFTSetupCreate(DFTConfig config) {
    DFTSetup setup = malloc(sizeof(struct DFTSetupStruct));
    setup->config = config;
#if APPLE
    setup->implementer = vDSP_DFT_zop_CreateSetup(NULL, config.nfft, config.forward ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE);
#else

#endif
    return setup;
}

void DFTPerform(DFTSetup setup, ComplexFloatSplit* input, ComplexFloatSplit* output){
#if APPLE
    vDSP_DFT_Execute(setup->implementer,
                     input->real_p, input->imag_p,
                     output->real_p, output->imag_p);
#else

#endif
}

void DFTSetupDestroy(DFTSetup setup) {
#if APPLE
    vDSP_DFT_DestroySetup(setup->implementer);
    free(setup);
#else

#endif
}

void SplitComplex(ComplexFloatSplit *split, ComplexFloat *complex, int size) {
#if APPLE
    void op_split_complex_fill(ComplexFloatSplit *split, ComplexFloat *cmplx, int size) {
    vDSP_ctoz((DSPComplex *)cmplx, 1, (DSPSplitComplex *) split, 2, size);
#else
    for (int i = 0; i < size; ++i){
        ComplexFloat item = complex[i];
        split->real_p[i] = item.real;
        split->imag_p[i] = item.imag;
    }
#endif
}


DFTConfig DFTConfigCreate(int nfft, bool forward, bool complex) {
    DFTConfig result;
    result.nfft = nfft;
    result.complex = complex;
    result.forward = forward;
    return result;
}
