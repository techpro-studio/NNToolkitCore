//
// Created by Alex on 19.11.2020.
//

#ifndef rnn_h
#define rnn_h

#include "activation.h"

typedef struct {
    float *W;
    float *U;
    float *b_i;
    float *b_h;
} RNNWeights;

typedef struct {
    int input_feature_channels;
    int output_feature_channels;
    bool v2;
    bool return_sequences;
    int timesteps;
    ActivationFunction activation;
} RNNConfig;

typedef struct {
    int mini_batch_size;
} RNNTrainingConfig;

typedef struct {
    float * d_W;
    float * d_U;
    float * d_b_i;
    float * d_b_h;
    float * d_X;
} RNNGradient;

typedef struct RNNStruct* RNN;


RNNConfig RNNConfigCreate(
    int input_feature_channels,
    int output_feature_channels,
    bool v2,
    bool return_sequences,
    int timesteps,
    ActivationFunction activation
);

RNNTrainingConfig RecurrentTrainingConfigCreate(int mini_batch_size);

RNNGradient * RNNGradientCreate(RNNConfig config, RNNTrainingConfig training_config);

RNNWeights* RNNGetWeights(RNN filter);

RNN RNNCreateForInference(RNNConfig config);

RNN RNNCreateForTraining(RNNConfig config, RNNTrainingConfig training_config);

int RNNApplyInference(RNN filter, const float *input, float* output);

int RNNApplyTrainingBatch(RNN filter, const float *input, float* output);

void RNNCalculateGradient(RNN filter, RNNGradient *gradients, float *d_out);

void RNNDestroy(RNN filter);

#endif //rnn_h
