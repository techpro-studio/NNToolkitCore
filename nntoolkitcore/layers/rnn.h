//
// Created by Alex on 19.11.2020.
//

#ifndef rnn_h
#define rnn_h

#include "activation.h"
#include "recurrent.h"

typedef RecurrentWeights RNNWeights;
typedef RecurrentGradient RNNGradient;
typedef RecurrentTrainingConfig RNNTrainingConfig;

typedef struct {
    int input_feature_channels;
    int output_feature_channels;
    bool v2;
    bool return_sequences;
    int timesteps;
    ActivationFunction activation;
} RNNConfig;

typedef struct RNNStruct* RNN;


RNNConfig RNNConfigCreate(
    int input_feature_channels,
    int output_feature_channels,
    bool v2,
    bool return_sequences,
    int timesteps,
    ActivationFunction activation
);

RNNWeights* RNNGetWeights(RNN filter);

RNNGradient * RNNGradientCreate(RNNConfig config, RNNTrainingConfig training_config);

RNN RNNCreateForInference(RNNConfig config);

RNN RNNCreateForTraining(RNNConfig config, RNNTrainingConfig training_config);

int RNNApplyInference(RNN filter, const float *input, float* output);

int RNNApplyTrainingBatch(RNN filter, const float *input, float* output);

void RNNCalculateGradient(RNN filter, RNNGradient *gradients, float *d_out);

void RNNDestroy(RNN filter);

#endif //rnn_h
