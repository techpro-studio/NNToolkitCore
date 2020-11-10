//
//  spectrogram.c
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "nn_toolkit_core/signal/spectrogram.h"
#include "nn_toolkit_core/signal/fft.h"
#include "nn_toolkit_core/core/ops.h"
#include "nn_toolkit_core/core/loop.h"
#include "stdlib.h"
#include "string.h"


typedef void (*spectrogram_implementer)(Spectrogram filter, const float* input, float* output);


struct SpectrogramStruct{
    SpectrogramConfig config;
    DFTSetup dft_setup;
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

inline static void magnitude(float* real_p, float* imag_p, float *freqs, const int size)
{
    op_vec_magnitudes(real_p, imag_p, freqs, size);
    op_vec_sqrt(freqs, freqs, size);
    op_vec_add_sc(freqs, 1.5849e-13f, freqs, size);
    op_vec_db(freqs, 1.0f, freqs, size);
}


static void real_spectrogram(Spectrogram filter, const float* input, float* output){
    P_LOOP_START(filter->config.ntime_series, timed)
        int nfft = filter->config.nfft;
        int nfreq = filter->config.nfreq;
        float norm_factor = filter->config.fft_normalization_factor;
        float input_re_im[nfft * 2];
        memset(input_re_im, 0, nfft * 2 * sizeof(float));
        float output_memory[2 * nfft];
        op_vec_mul(filter->window, input + timed * filter->config.step, input_re_im, nfft);
        complex_float_spl input_split = { input_re_im, input_re_im + nfft};
        complex_float_spl output_split = { output_memory, output_memory + nfft};
        DFTPerform(filter->dft_setup, &input_split , &output_split);
        op_vec_mul_sc(output_memory, norm_factor, output_memory, nfft * 2);
        magnitude(output_memory, output_memory + nfft, output + timed * nfreq, nfreq);
    P_LOOP_END
}

Spectrogram SpectrogramCreate(SpectrogramConfig config){
    Spectrogram filter = malloc(sizeof(struct SpectrogramStruct));
    filter->config = config;
    filter->dft_setup = DFTSetupCreate(DFTConfigCreate(config.nfft, true, false));
    filter->window = malloc(config.nfft * sizeof(float));
    for (int i = 0; i < config.nfft; ++i)
        filter->window[i] = 1.0f;
    return filter;
}
void SpectrogramSetWindowFunc(Spectrogram filter, window_fn fn) {
    fn(filter->window, filter->config.nfft);
}

void complex_spectrogram(Spectrogram filter, const float* input, float* output) {
    P_LOOP_START(filter->config.ntime_series, timed)
        int nfft = filter->config.nfft;
        int nfreq = filter->config.nfreq;
        float output_memory[nfft * 2];
        float input_memory[nfft * 2];
        complex_float_spl input_split = {input_memory, input_memory + nfft};
        op_split_complex_fill(&input_split, ((complex_float *)input) + timed * filter->config.step, nfft);
        complex_float_spl output_split = { output_memory, output_memory + nfft};
        op_vec_mul(filter->window, input_split.real_p, input_split.real_p, nfft);
        op_vec_mul(filter->window, input_split.imag_p, input_split.imag_p, nfft);
        DFTPerform(filter->dft_setup, &input_split , &output_split);
        magnitude(output_memory, output_memory + nfft, output + timed * nfreq, nfreq);
    P_LOOP_END
}

void SpectrogramApply(Spectrogram filter, const float *input, float* output){
    spectrogram_implementer impl = filter->config.complex ? complex_spectrogram : real_spectrogram;
    impl(filter, input, output);
}

void SpectrogramDestroy(Spectrogram filter){
    DFTSetupDestroy(filter->dft_setup);
    free(filter);
}

