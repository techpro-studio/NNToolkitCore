//
// Created by Alex on 15.12.2020.
//

#include "mel_filterbank.h"
#include "nntoolkitcore/core/ops.h"
#include "stdlib.h"
#include "nntoolkitcore/core/memory.h"


static void hertz_to_mel(const float *hz, float *mel, int size){
    op_vec_div_sc(hz, 700.0f, mel, size);
    op_vec_add_sc(mel, 1.0, mel, size);
    op_vec_log(mel, mel, size);
    op_vec_mul_sc(mel, 1127.0f, mel, size);
}

static void mel_to_hertz(const float *mel, float *hz, int size){
    op_vec_div_sc(mel, 1127.0f, hz, size);
    op_vec_exp(hz, hz, size);
    op_vec_add_sc(hz, -1.0, hz, size);
    op_vec_mul_sc(hz, 700, hz, size);
}

struct MelFilterBankStruct{
    MelFilterBankConfig config;

    float *weights;
    float *fft_freqs;
    float *mel_freqs;
};

MelFilterBankConfig MelFilterBankConfigCreate(int n_mels, int n_fft, int sample_rate, float lower_hz, float upper_hz) {
    MelFilterBankConfig config;
    config.lower_hz = lower_hz;
    config.upper_hz = upper_hz;
    config.sample_rate = sample_rate;
    config.n_fft = n_fft;
    config.n_mels = n_mels;
    return config;
}

static float* init_mel_freqs(MelFilterBankConfig config){
    float *band = f_malloc(config.n_mels + 2);

    float edges[2] = {config.lower_hz, config.upper_hz};

    hertz_to_mel(edges, edges, 2);

    float step = (edges[1] - edges[0]) / (config.n_mels + 1);


    for (int i = 0; i < config.n_mels + 2; i++){
        band[i] = edges[0] + (step * i);
    }


    mel_to_hertz(band, band, config.n_mels + 2);

    return band;
}

static float * init_fft_freqs(MelFilterBankConfig config) {
    int n_bins = config.n_fft / 2 + 1;
    float *bin_hz = f_malloc(n_bins);
    float step = config.sample_rate / config.n_fft;
    for (int i = 0; i < n_bins; ++i){
        bin_hz[i] = step * i;
    }
    return bin_hz;
}

#include "nntoolkitcore/core/debug.h"

static void init_default_filter_bank(MelFilterBank bank){
    int n_mels = bank->config.n_mels;
    int n_bins = bank->config.n_fft / 2 + 1;
    float *lower_slope = f_malloc(n_bins);
    float *upper_slope = f_malloc(n_bins);



    for (int i = 0; i < n_mels; ++i){
        float lower_edge_mel = bank->mel_freqs[i];
        float center_mel = bank->mel_freqs[i + 1];
        float upper_edge_mel = bank->mel_freqs[i + 2];



        op_vec_add_sc(bank->fft_freqs,  -1.0f * lower_edge_mel, lower_slope, n_bins);
        op_vec_div_sc(lower_slope, center_mel - lower_edge_mel, lower_slope, n_bins);


        op_vec_neg(bank->fft_freqs, upper_slope, n_bins);

        op_vec_add_sc(upper_slope, upper_edge_mel, upper_slope, n_bins);
        op_vec_div_sc(upper_slope, upper_edge_mel - center_mel, upper_slope, n_bins);

        float* result = bank->weights + i * n_bins;

        op_vec_min(upper_slope, lower_slope, result, n_bins);
        op_vec_max_sc(result, 0.0, result, n_bins);

        result[0] = 0.0f;
    }

    float bank_tr[n_mels * n_bins];

    op_mat_transp(bank->weights, bank_tr, n_bins, n_mels);

    print_matrix(bank_tr, n_bins, n_mels);

    free(lower_slope);
    free(upper_slope);
}

MelFilterBank MelFilterBankCreate(MelFilterBankConfig config) {
    MelFilterBank filter_bank = malloc(sizeof(struct MelFilterBankStruct));
    filter_bank->config = config;
    int hz_bins = (config.n_fft / 2 + 1);
    filter_bank->weights = f_malloc(hz_bins * config.n_mels);
    filter_bank->mel_freqs = init_mel_freqs(config);
    filter_bank->fft_freqs = init_fft_freqs(config);
    init_default_filter_bank(filter_bank);
    return filter_bank;
}


void MelFilterBankApply(MelFilterBank filter_bank, const float* spectrogram, float *mel_spectrogram, int timesteps) {
    op_mat_mul(spectrogram, filter_bank->weights, mel_spectrogram, timesteps, filter_bank->config.n_mels, (filter_bank->config.n_fft / 2 + 1));
}

void MelFilterBankDestroy(MelFilterBank bank) {
    free(bank->fft_freqs);
    free(bank->weights);
    free(bank->mel_freqs);
    free(bank);
}
