//
// Created by Alex on 10.11.2020.
//

#include "nntoolkitcore/signal/dft.h"
#include "stdlib.h"
#include "nntoolkitcore/core/loop.h"

#define USE_APPLE_FFT APPLE

#if USE_APPLE_FFT
    #include <Accelerate/Accelerate.h>
#else
    #include "third_party/kissfft/kiss_fft.h"
#endif


struct DFTSetupStruct{
    DFTConfig config;
    void *implementer;
};

DFTSetup DFTSetupCreate(DFTConfig config) {
    DFTSetup setup = malloc(sizeof(struct DFTSetupStruct));
    setup->config = config;
#if USE_APPLE_FFT
    setup->implementer = vDSP_DFT_zop_CreateSetup(NULL, config.nfft, config.forward ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE);
#else
    setup->implementer = kiss_fft_alloc(config.nfft, (int)!config.forward, 0, 0);
#endif
    return setup;
}

void DFTPerform(DFTSetup setup, ComplexFloatSplit* input, ComplexFloatSplit* output){
#if USE_APPLE_FFT
    vDSP_DFT_Execute(setup->implementer,
                     input->real_p, input->imag_p,
                     output->real_p, output->imag_p);
#else
    int nfft = setup->config.nfft;
    kiss_fft_cpx in[nfft];
    kiss_fft_cpx out[nfft];
    join_complex_split((ComplexFloat *)in, input, nfft);
    kiss_fft(setup->implementer, in, out);
    split_complex(h, (ComplexFloat *) out, nfft);

#endif
}

void DFTSetupDestroy(DFTSetup setup) {
#if USE_APPLE_FFT
    vDSP_DFT_DestroySetup(setup->implementer);
    free(setup);
#else
    kiss_fft_free(setup->implementer);
    free(setup);
#endif
}

void split_complex(ComplexFloatSplit *split, ComplexFloat *complex, int size) {
#if USE_APPLE_FFT
    vDSP_ctoz((DSPComplex *)complex, 1, (DSPSplitComplex *) split, 2, size);
#else
    P_LOOP_START(size, i)
        ComplexFloat item = complex[i];
        split->real_p[i] = item.real;
        split->imag_p[i] = item.imag;
    P_LOOP_END
#endif
}

DFTConfig DFTConfigCreate(int nfft, bool forward, bool complex) {
    DFTConfig result;
    result.nfft = nfft;
    result.complex = complex;
    result.forward = forward;
    return result;
}

void join_complex_split(ComplexFloat *complex, ComplexFloatSplit *split, int size) {
    P_LOOP_START(size, i)
        ComplexFloat c = { split->real_p[i], split->imag_p[i]};
        complex[i] = c;
    P_LOOP_END
}
