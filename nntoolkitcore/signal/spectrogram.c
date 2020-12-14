//
//  spectrogram.c
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "nntoolkitcore/signal/spectrogram.h"
#include "nntoolkitcore/signal/dft.h"
#include "nntoolkitcore/core/ops.h"
#include "nntoolkitcore/core/loop.h"
#include "nntoolkitcore/core/memory.h"
#include "stdlib.h"



typedef void(*spectrogram_mode_run)(void *params, float* real_p, float* imag_p, float *freqs, const int size,  const float window_scale_factor);


typedef float(*spectrogram_mode_calculate_window_factor) (float *window, int size);


struct SpectrogramModeStruct {
    void *params;
    void *buffer;
    float window_scale_factor;
    spectrogram_mode_calculate_window_factor win_f_calc;
    spectrogram_mode_run run_fn;
};


SpectrogramMode create_spectrogram_mode(spectrogram_mode_run run_fn, spectrogram_mode_calculate_window_factor win_f_calc){
    SpectrogramMode mode = malloc(sizeof(SpectrogramMode));
    mode->params = NULL;
    mode->run_fn = run_fn;
    mode->window_scale_factor = 0.0f;
    return mode;
}

void spectrogram_mode_destroy(SpectrogramMode mode){
    if (mode->params){
        free(mode->params);
    }
    free(mode);
}


static void magnitude(void* params, float* real_p, float* imag_p, float *freqs, const int size, const float window_scale_factor)
{
    op_vec_magnitudes(real_p, imag_p, freqs, size);
    op_vec_sqrt(freqs, freqs, size);
    op_vec_div_sc(freqs, window_scale_factor, freqs, size);
}

static float magnitude_calc_win_factor(float *window, int size) {
    float result;
    op_vec_sum(window, &result, size);
    return result;
}

SpectrogramMode SpectrogramModeCreateMagnitude(){
    return create_spectrogram_mode(magnitude, magnitude_calc_win_factor);
}


static void psd(void *params, float* real_p, float* imag_p, float *freqs, const int size,  const float window_scale_factor)
{
    op_vec_magnitudes(real_p, imag_p, freqs, size);
    op_vec_div_sc(freqs + 1, 2 * window_scale_factor, freqs, size - 1);
    freqs[0] = freqs[0] * window_scale_factor;
    freqs[size - 1] = freqs[size - 1] * window_scale_factor;
}

static float psd_calc_win_factor(float *window, int size) {
    float result;
    float buffer[size];
    op_vec_mul(window, window, buffer, size);
    op_vec_sum(buffer, &result, size);
    return result;
}

SpectrogramMode SpectrogramModeCreatePSD(int fs){
    SpectrogramMode mode = create_spectrogram_mode(psd, psd_calc_win_factor);
    mode->params = malloc(sizeof(int));
    *((int *)mode->params) = fs;
    return mode;
}

struct SpectrogramStruct{
    SpectrogramConfig config;
    DFTSetup dft_setup;
    SpectrogramMode mode;
    float *window;
};

typedef void (*spectrogram_implementer)(Spectrogram filter, const float* input, float* output);

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

static void real_spectrogram(Spectrogram filter, const float* input, float* output){
    P_LOOP_START(filter->config.ntime_series, timed)
        int nfft = filter->config.nfft;
        int nfreq = filter->config.nfreq;
        float norm_factor = filter->config.fft_normalization_factor;
        float input_re_im[nfft * 2];
        f_zero(input_re_im, nfft * 2);
        float output_memory[2 * nfft];
        op_vec_mul(filter->window, input + timed * filter->config.step, input_re_im, nfft);
        ComplexFloatSplit input_split = {input_re_im, input_re_im + nfft};
        ComplexFloatSplit output_split = {output_memory, output_memory + nfft};
        DFTPerform(filter->dft_setup, &input_split , &output_split);
        op_vec_mul_sc(output_memory, norm_factor, output_memory, nfft * 2);
        filter->mode->run_fn(filter->mode->params, output_memory, output_memory + nfft, output + timed * nfreq, nfreq, filter->mode->window_scale_factor);
    P_LOOP_END
}

Spectrogram SpectrogramCreate(SpectrogramConfig config, SpectrogramMode mode){
    Spectrogram filter = malloc(sizeof(struct SpectrogramStruct));
    filter->config = config;
    filter->mode = mode;
    filter->dft_setup = DFTSetupCreate(DFTConfigCreate(config.nfft, true, false));
    filter->window = malloc(config.nfft * sizeof(float));
    for (int i = 0; i < config.nfft; ++i)
        filter->window[i] = 1.0f;
    filter->mode->window_scale_factor = filter->mode->win_f_calc(filter->window, filter->mode->buffer, filter->config.nfft);
    return filter;
}
void SpectrogramSetWindowFunc(Spectrogram filter, window_fn fn) {
    fn(filter->window, filter->config.nfft);
    filter->mode->window_scale_factor
     = filter->mode->win_f_calc(filter->window, filter->mode->buffer, filter->config.nfft);
}

void complex_spectrogram(Spectrogram filter, const float* input, float* output) {
    P_LOOP_START(filter->config.ntime_series, timed)
        int nfft = filter->config.nfft;
        int nfreq = filter->config.nfreq;
        float output_memory[nfft * 2];
        float input_memory[nfft * 2];
        ComplexFloatSplit input_split = {input_memory, input_memory + nfft};
        split_complex(((ComplexFloat *) input) + timed * filter->config.step, &input_split , nfft);
        ComplexFloatSplit output_split = {output_memory, output_memory + nfft};
        op_vec_mul(filter->window, input_split.real_p, input_split.real_p, nfft);
        op_vec_mul(filter->window, input_split.imag_p, input_split.imag_p, nfft);
        DFTPerform(filter->dft_setup, &input_split , &output_split);
        filter->mode->run_fn(filter->mode->params, output_memory, output_memory + nfft, output + timed * nfreq, nfreq, filter->mode->window_scale_factor);
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

