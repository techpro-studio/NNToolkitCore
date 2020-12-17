//
// Created by Alex on 15.12.2020.
//

#ifndef mel_filterbank_h
#define mel_filterbank_h


#if defined __cplusplus
extern "C" {
#endif


typedef struct {
    int n_mels;
    int n_fft;
    int sample_rate;
    float lower_hz;
    float upper_hz;
} MelFilterBankConfig;


MelFilterBankConfig
MelFilterBankConfigCreate(
      int n_mels,
      int n_fft,
      int sample_rate,
      float lower_hz,
      float upper_hz
);

typedef struct MelFilterBankStruct* MelFilterBank;

MelFilterBank MelFilterBankCreate(MelFilterBankConfig config);

void MelFilterBankApply(MelFilterBank filter_bank, const float* spectrogram, float *mel_spectrogram, int timesteps);

void MelFilterBankDestroy(MelFilterBank filter_bank);

#if defined __cplusplus
}
#endif

#endif //mel_filterbank_h
