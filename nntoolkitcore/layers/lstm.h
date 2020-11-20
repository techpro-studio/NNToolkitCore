//
//  Header.h
//  
//
//  Created by Alex on 01.10.2020.
//

#ifndef lstm_h
#define lstm_h

#include <stdio.h>
#include "stdbool.h"
#include "activation.h"
#include "recurrent_base.h"

#if defined __cplusplus
extern "C" {
#endif

typedef RecurrentWeights LSTMWeights;
typedef RecurrentGradient LSTMGradient;
typedef RecurrentTrainingConfig LSTMTrainingConfig;

typedef struct {
    ActivationFunction candidate_gate_activation;
    ActivationFunction input_gate_activation;
    ActivationFunction forget_gate_activation;
    ActivationFunction output_gate_activation;
    ActivationFunction output_activation;
} LSTMActivations;

LSTMActivations LSTMActivationsCreate(
    ActivationFunction input_gate_activation,
    ActivationFunction forget_gate_activation,
    ActivationFunction candidate_gate_activation,
    ActivationFunction output_gate_activation,
    ActivationFunction output_activation
);

LSTMActivations LSTMActivationsCreateDefault(int size);

void LSTMActivationsDestroy(LSTMActivations activations);

typedef struct {
    int input_feature_channels;
    int output_feature_channels;
    bool v2;
    bool return_sequences;
    int timesteps;
    LSTMActivations activations;
} LSTMConfig;

LSTMConfig LSTMConfigCreate(
    int input_feature_channels,
    int output_feature_channels,
    bool v2, bool return_sequences,
    int timesteps,
    LSTMActivations activations
);

typedef struct LSTMStruct* LSTM;

LSTMGradient * LSTMGradientCreate(LSTMConfig config, LSTMTrainingConfig training_config);

LSTMWeights* LSTMGetWeights(LSTM filter);

LSTM LSTMCreateForInference(LSTMConfig config);

LSTM LSTMCreateForTraining(LSTMConfig config, LSTMTrainingConfig training_config);

int LSTMApplyInference(LSTM filter, const float *input, float* output);

int LSTMApplyTrainingBatch(LSTM filter, const float *input, float* output);

void LSTMCalculateGradient(LSTM filter, LSTMGradient *gradients, float *d_out);

void LSTMDestroy(LSTM filter);

#if defined __cplusplus
}
#endif

#endif /* Header_h */
