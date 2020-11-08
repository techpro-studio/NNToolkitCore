//
//  spectrogram.c
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "spectrogram.h"
#include <Accelerate/Accelerate.h>
#include "operations.h"
#include "loops.h"


typedef void (*spectrogram_implementer)(Spectrogram filter, const float* input, float* output);


struct SpectrogramStruct{
    SpectrogramConfig config;
    void* fft_setup;
    float *window;
};

SpectrogramConfig SpectrogramConfigCreate(int nfft, int noverlap, int input_size, bool complex, float fft_normalization_factor){
    SpectrogramConfig config;
    config.nfft = nfft;
    config.noverlap = noverlap;
    config.input_size = input_size;
    config.step = nfft - noverlap;
    config.complex = complex;
    config.nfreq = complex ? nfft : nfft / 2 + 1;
    config.ntime_series = (input_size - noverlap) / config.step;
    config.fft_normalization_factor = fft_normalization_factor;
    return config;
}

inline static void magnitude(DSPSplitComplex *split, float *freqsPtr, const int vectorSize)
{
    vDSP_zvmags(split, 1, freqsPtr, 1, vectorSize);
    vvsqrtf(freqsPtr, freqsPtr, &vectorSize);
    const float k_adjust_0DB = 1.5849e-13f;
    vDSP_vsadd(freqsPtr, 1, &k_adjust_0DB, freqsPtr, 1, vectorSize);
    Float32 one = 1;
    vDSP_vdbcon(freqsPtr, 1, &one, freqsPtr, 1, vectorSize, 0);
}


static void real_spectrogram(Spectrogram filter, const float* input, float* output){
    P_LOOP_START(filter->config.ntime_series, timed)
        int nfft = filter->config.nfft;
        int nfreq = filter->config.nfreq;
        float norm_factor = filter->config.fft_normalization_factor;
        float input_re_im[nfft * 2];
        memset(input_re_im, 0, nfft * 2 * sizeof(float));
        float output_memory[2 * nfft];
        DSPSplitComplex output_split = {output_memory, output_memory + nfft};
        vDSP_vmul(filter->window, 1, input + timed * filter->config.step, 1, input_re_im, 1, nfft);
        vDSP_DFT_Execute(filter->fft_setup,
                         input_re_im, input_re_im + nfft,
                         output_split.realp, output_split.imagp);
        vDSP_vsmul(output_memory, 1, &norm_factor, output_memory, 1, nfft * 2);
                magnitude(&output_split, output + timed * nfreq, nfreq);
    P_LOOP_END
}

Spectrogram SpectrogramCreate(SpectrogramConfig config){
    Spectrogram filter = malloc(sizeof(struct SpectrogramStruct));
    filter->config = config;
    filter->fft_setup = vDSP_DFT_zop_CreateSetup(NULL, config.nfft, vDSP_DFT_FORWARD);
    filter->window = malloc(config.nfft * sizeof(float));
    for (int i = 0; i < config.nfft; ++i)
        filter->window[i] = 1.0;
    return filter;
}
void SpectrogramSetWindowFunc(Spectrogram filter, window_fn fn) {
    fn(filter->window, filter->config.nfft);
}

void complex_spectrogram(Spectrogram filter, const float* input, float* output) {
    dispatch_apply(filter->config.ntime_series, DISPATCH_APPLY_AUTO, ^(size_t timed) {
        int nfft = filter->config.nfft;
        int nfreq = filter->config.nfreq;
        float output_memory[nfft * 2];
        float input_memory[nfft * 2];
        DSPSplitComplex input_split = {input_memory, input_memory + nfft};
        DSPSplitComplex output_split = {output_memory, output_memory + nfft};
        vDSP_ctoz(((DSPComplex *)input) + timed * filter->config.step, 2, &input_split, 1, nfft);
        vDSP_vmul(filter->window, 1, input_split.realp, 1, input_split.realp, 1, nfft);
        vDSP_vmul(filter->window, 1, input_split.imagp, 1, input_split.imagp, 1, nfft);
        vDSP_DFT_Execute(filter->fft_setup,
                         input_split.realp, input_split.imagp,
                         output_split.realp, output_split.imagp);
        magnitude(&output_split, output + timed * nfreq, nfreq);
    });
}

void SpectrogramApply(Spectrogram filter, const float *input, float* output){
    spectrogram_implementer impl = filter->config.complex ? complex_spectrogram : real_spectrogram;
    impl(filter, input, output);
}

void SpectrogramDestroy(Spectrogram filter){
    vDSP_DFT_DestroySetup(filter->fft_setup);
    free(filter);
}

