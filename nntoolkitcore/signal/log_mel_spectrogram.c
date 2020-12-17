//
// Created by Alex on 18.12.2020.
//

#include "log_mel_spectrogram.h"
#include "stdlib.h"
#include "nntoolkitcore/core/memory.h"
#include "nntoolkitcore/core/ops.h"

struct LogMelSpectrogramStruct {
    MelFilterBankConfig mel_filter_bank_config;
    Spectrogram spectrogram;
    MelFilterBank bank;
    int ts;
    int mel_output_size;
    float *buffer;
};

LogMelSpectrogram LogMelSpectrogramCreate(Spectrogram spectrogram, MelFilterBankConfig mel_filter_bank_config){
    LogMelSpectrogram filter = malloc(sizeof(struct LogMelSpectrogramStruct));
    filter->spectrogram = spectrogram;
    SpectrogramConfig s_cfg = SpectrogramGetConfig(spectrogram);
    filter->ts = s_cfg.ntime_series;
    filter->mel_output_size = filter->ts * mel_filter_bank_config.n_mels;
    filter->mel_filter_bank_config = mel_filter_bank_config;
    filter->bank = MelFilterBankCreate(mel_filter_bank_config);
    filter->buffer = f_malloc(s_cfg.ntime_series * s_cfg.nfreq);
    return filter;
}

void LogMelSpectrogramApply(LogMelSpectrogram filter, const float *input, float* output){
    SpectrogramApply(filter->spectrogram, input, filter->buffer);
    MelFilterBankApply(filter->bank, filter->buffer, output, filter->ts);
    op_vec_add_sc(output, 1.5849e-13, output, filter->mel_output_size);
    op_vec_log(output, output, filter->mel_output_size);
}

void LogMelSpectrogramDestroy(LogMelSpectrogram filter){
    MelFilterBankDestroy(filter->bank);
    free(filter->buffer);
    free(filter);
}
