//
//  spectrogram.c
//  mac_test
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "spectrogram.h"
#include <dispatch/dispatch.h>

typedef void (*spectrogram_implementer)(SpectrogramFilter *filter, const float* input, float* output);


SpectrogramConfig SpectrogramConfigCreate(int nfft, int noverlap, int inputSize, bool complex, float fftNormalizationFactor){
    SpectrogramConfig config;
    config.nfft = nfft;
    config.noverlap = noverlap;
    config.inputSize = inputSize;
    config.step = nfft - noverlap;
    config.complex = complex;
    config.nfreq = complex ? nfft : nfft / 2 + 1;
    config.ntimeSeries = (inputSize - noverlap) / config.step;
    config.fftNormalizationFactor = fftNormalizationFactor;
    return config;
}

inline static void Magnitude(DSPSplitComplex *split, float *freqsPtr, const int vectorSize)
{
    vDSP_zvmags(split, 1, freqsPtr, 1, vectorSize);
    vvsqrtf(freqsPtr, freqsPtr, &vectorSize);
    const Float32 kAdjust0DB = 1.5849e-13;
    vDSP_vsadd(freqsPtr, 1, &kAdjust0DB, freqsPtr, 1, vectorSize);
    Float32 one = 1;
    vDSP_vdbcon(freqsPtr, 1, &one, freqsPtr, 1, vectorSize, 0);
}


static void real_spectrogram(SpectrogramFilter *filter, const float* input, float* output){
    dispatch_apply(filter->config.ntimeSeries, DISPATCH_APPLY_AUTO, ^(size_t timed) {
        int nfft = filter->config.nfft;
        int nfreq = filter->config.nfreq;
        float normFactor = filter->config.fftNormalizationFactor;
        float inputReIm[nfft * 2];
        memset(inputReIm, 0, nfft * 2 * sizeof(float));
        float outputMemory[2 * nfft];
        DSPSplitComplex outputSplit = {outputMemory, outputMemory + nfft};
        vDSP_vmul(filter->window, 1, input + timed * filter->config.step, 1, inputReIm, 1, nfft);
        vDSP_DFT_Execute(filter->fftSetup,
                         inputReIm, inputReIm + nfft,
                         outputSplit.realp, outputSplit.imagp);
        vDSP_vsmul(outputMemory, 1, &normFactor, outputMemory, 1, nfft * 2);
        Magnitude(&outputSplit, output + timed * nfreq, nfreq);
    });

}

SpectrogramFilter* SpectrogramFilterCreate(SpectrogramConfig config){
    SpectrogramFilter* filter = malloc(sizeof(SpectrogramFilter));
    filter->config = config;
    filter->fftSetup = vDSP_DFT_zop_CreateSetup(NULL, config.nfft, vDSP_DFT_FORWARD);
    filter->window = malloc(config.nfft * sizeof(float));
    for (int i = 0; i < config.nfft; ++i)
        filter->window[i] = 1.0;
    return filter;
}

void SpectrogramFilterSetWindowType(SpectrogramFilter *filter, FFTWindowType type) {
    switch (type) {
        case FFTWindowTypeHann:
            vDSP_hann_window(filter->window, filter->config
                             .nfft, 0);
            break;
        case FFTWindowTypeHamming:
            vDSP_hamm_window(filter->window, filter->config.nfft, 0);
            break;
        case FFTWindowTypeBlackMan:
            vDSP_blkman_window(filter->window, filter->config.nfft, 0);
        default:
            break;
    }
}

void complex_spectrogram(SpectrogramFilter *filter, const float* input, float* output) {
    dispatch_apply(filter->config.ntimeSeries, DISPATCH_APPLY_AUTO, ^(size_t timed) {
        int nfft = filter->config.nfft;
        int nfreq = filter->config.nfreq;
        float outputMemory[nfft * 2];
        float inputMemory[nfft * 2];
        DSPSplitComplex inputSplit = {inputMemory, inputMemory + nfft};
        DSPSplitComplex outputSplit = {outputMemory, outputMemory + nfft};
        vDSP_ctoz(((DSPComplex *)input) + timed * filter->config.step, 2, &inputSplit, 1, nfft);
        vDSP_vmul(filter->window, 1, inputSplit.realp, 1, inputSplit.realp, 1, nfft);
        vDSP_vmul(filter->window, 1, inputSplit.imagp, 1, inputSplit.imagp, 1, nfft);
        vDSP_DFT_Execute(filter->fftSetup,
                         inputSplit.realp, inputSplit.imagp,
                         outputSplit.realp, outputSplit.imagp);
        Magnitude(&outputSplit, output + timed * nfreq, nfreq);
    });
}

void SpectrogramFilterApply(SpectrogramFilter *filter, const float *input, float* output){
    spectrogram_implementer impl = filter->config.complex ? complex_spectrogram : real_spectrogram;
    impl(filter, input, output);
}

void SpectrogramFilterDestroy(SpectrogramFilter *filter){
    vDSP_DFT_DestroySetup(filter->fftSetup);
    free(filter);
}

