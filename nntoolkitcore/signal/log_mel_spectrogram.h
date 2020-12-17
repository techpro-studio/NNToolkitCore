//
// Created by Alex on 18.12.2020.
//

#ifndef log_mel_spectrogram_h
#define log_mel_spectrogram_h

#include "spectrogram.h"
#include "mel_filterbank.h"


typedef struct LogMelSpectrogramStruct* LogMelSpectrogram;

LogMelSpectrogram LogMelSpectrogramCreate(Spectrogram spectrogram, MelFilterBankConfig mel_filter_bank_config);

void LogMelSpectrogramApply(LogMelSpectrogram filter, const float *input, float* output);

void LogMelSpectrogramDestroy(LogMelSpectrogram filter);

#endif //log_mel_spectrogram_h
