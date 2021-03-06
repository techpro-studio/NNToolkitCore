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
#include "recurrent.h"

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
    RecurrentConfig base;
    bool v2;
    LSTMActivations activations;
} LSTMConfig;

LSTMConfig LSTMConfigCreate(
    int input_feature_channels,
    int output_feature_channels,
    bool return_sequences,
    int timesteps,
    bool v2,
    LSTMActivations activations
);

typedef struct LSTMStruct* LSTM;

LSTMWeights* LSTMGetWeights(LSTM filter);

LSTM LSTMCreateForInference(LSTMConfig config);

int LSTMApplyInference(LSTM filter, const float *input, float* output);

LSTM LSTMCreateForTraining(LSTMConfig config, LSTMTrainingConfig training_config);

LSTMGradient * LSTMGradientCreate(LSTMConfig config, LSTMTrainingConfig training_config);

int LSTMApplyTrainingBatch(LSTM filter, const float *input, float* output);

void LSTMCalculateGradient(LSTM filter, LSTMGradient *gradient, float *d_out);

void LSTMDestroy(LSTM filter);

#if defined __cplusplus
}
#endif

#endif /* Header_h */
