//
// Created by Alex on 19.11.2020.
//

#ifndef rnn_h
#define rnn_h

#include "activation.h"
#include "recurrent.h"

#if defined __cplusplus
extern "C" {
#endif

typedef RecurrentWeights RNNWeights;
typedef RecurrentGradient RNNGradient;
typedef RecurrentTrainingConfig RNNTrainingConfig;

typedef struct {
    RecurrentConfig base;
    bool v2;
    ActivationFunction activation;
} RNNConfig;

typedef struct RNNStruct *RNN;

RNNConfig RNNConfigCreate(
    int input_feature_channels,
    int output_feature_channels,
    bool return_sequences,
    int timesteps,
    bool v2,
    ActivationFunction activation
);

RNNWeights *RNNGetWeights(RNN filter);

RNN RNNCreateForInference(RNNConfig config);

int RNNApplyInference(RNN filter, const float *input, float *output);

RNN RNNCreateForTraining(RNNConfig config, RNNTrainingConfig training_config);

RNNGradient *RNNGradientCreate(RNNConfig config, RNNTrainingConfig training_config);

int RNNApplyTrainingBatch(RNN filter, const float *input, float *output);

void RNNCalculateGradient(RNN filter, RNNGradient *gradients, float *d_out);

void RNNDestroy(RNN filter);

#if defined __cplusplus
}
#endif

#endif //rnn_h
