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


typedef void(*spectrogram_finish)(Spectrogram filter, float* real_p, float* imag_p, float *out);
typedef void(*spectrogram_calculate_factor) (Spectrogram filter);

struct SpectrogramStruct{
    SpectrogramConfig config;
    DFTSetup dft_setup;
    float *window;
    void  *params;
    float scale_factor;
    spectrogram_calculate_factor factor_calc;
    spectrogram_finish finish_fn;
};

static void magnitude(Spectrogram filter, float* real_p, float* imag_p, float *out) {
    int size = filter->config.nfreq;
    op_vec_magn_sq(real_p, imag_p, out, size);
    op_vec_sqrt(out, out, size);
    op_vec_div_sc(out, filter->scale_factor, out, size);
}

static void magnitude_calc_factor(Spectrogram filter) {
    op_vec_sum(filter->window, &(filter->scale_factor), filter->config.window_size);
}

static void psd(Spectrogram filter, float* real_p, float* imag_p, float *out) {
    int size = filter->config.nfreq;
    op_vec_magn_sq(real_p, imag_p, out, size);
    op_vec_div_sc(out + 1, 2 * filter->scale_factor, out, size - 1);
    out[0] = out[0] / filter->scale_factor;
    out[size - 1] = out[size - 1] / filter->scale_factor;
}

static void psd_calc_factor(Spectrogram filter) {
    int fs = ((PSDConfig *)filter->params)->fs;
    int size = filter->config.window_size;
    float buffer[size];
    float result = 0.0f;
    op_vec_mul(filter->window, filter->window, buffer, size);
    op_vec_sum(buffer, &result, size);
    filter->scale_factor = result * fs;
}

SpectrogramConfig SpectrogramConfigCreate(int nfft, int window_size, int noverlap, int input_size, float fft_normalization_factor){
    SpectrogramConfig config;
    config.nfft = nfft;
    config.window_size = window_size;
    config.noverlap = noverlap;
    config.input_size = input_size;
    config.step = window_size - noverlap;
    config.nfreq = nfft / 2 + 1;
    config.ntime_series = (input_size - noverlap) / config.step;
    config.fft_normalization_factor = fft_normalization_factor;
    return config;
}

Spectrogram spectrogram_create(SpectrogramConfig config, spectrogram_finish finish_fn, spectrogram_calculate_factor calc_fn){
    Spectrogram filter = malloc(sizeof(struct SpectrogramStruct));
    filter->config = config;
    filter->finish_fn = finish_fn;
    filter->factor_calc = calc_fn;
    filter->params = NULL;
    filter->dft_setup = DFTSetupCreate(DFTConfigCreate(config.nfft, true, false));
    filter->window = f_malloc(config.window_size);
    return filter;
}

Spectrogram SpectrogramCreatePSD(SpectrogramConfig config, PSDConfig psd_config){
    Spectrogram spectrogram = spectrogram_create(config, psd, psd_calc_factor);
    PSDConfig* cfg = malloc(sizeof(PSDConfig));
    *cfg = psd_config;
    spectrogram->params = cfg;
    SpectrogramSetWindowFunc(spectrogram, ones);
    return spectrogram;
}

Spectrogram SpectrogramCreateMagnitude(SpectrogramConfig config) {
    Spectrogram spectrogram = spectrogram_create(config, magnitude, magnitude_calc_factor);
    spectrogram->finish_fn = magnitude;
    spectrogram->factor_calc = magnitude_calc_factor;
    SpectrogramSetWindowFunc(spectrogram, ones);
    return spectrogram;
}

void SpectrogramSetScaleFactor(Spectrogram filter, float factor) {
    filter->scale_factor = factor;
}

void SpectrogramSetWindowFunc(Spectrogram filter, window_fn fn) {
    fn(filter->window, filter->config.window_size);
    filter->factor_calc(filter);
}

void SpectrogramApply(Spectrogram filter, const float *input, float* output){
    P_LOOP_START(filter->config.ntime_series, timed)
        int win_size = filter->config.window_size;
        int nfft = filter->config.nfft;
        int nfreq = filter->config.nfreq;

        float norm_factor = filter->config.fft_normalization_factor;
        float input_re_im[nfft * 2];
        f_zero(input_re_im, nfft * 2);
        float output_memory[2 * nfft];

        op_vec_mul(filter->window, input + timed * filter->config.step, input_re_im, win_size);

        ComplexFloatSplit input_split = {input_re_im, input_re_im + nfft};
        ComplexFloatSplit output_split = {output_memory, output_memory + nfft};
        DFTPerform(filter->dft_setup, &input_split , &output_split);

        if(norm_factor != 1.0f)
            op_vec_mul_sc(output_memory, norm_factor, output_memory, nfft * 2);

        filter->finish_fn(filter, output_memory, output_memory + nfft, output + timed * nfreq);
    P_LOOP_END
}

void SpectrogramDestroy(Spectrogram filter){
    DFTSetupDestroy(filter->dft_setup);
    free(filter);
}

