//
//  gru.h
//  audio_test
//
//  Created by Alex on 21.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef gru2_h
#define gru2_h

#include <stdio.h>
#include "stdbool.h"
#include "activation.h"
#include "recurrent.h"


#if defined __cplusplus
extern "C" {
#endif

typedef RecurrentWeights GRUWeights;
typedef RecurrentGradient GRUGradient;
typedef RecurrentTrainingConfig GRUTrainingConfig;

typedef struct {
    ActivationFunction z_gate_activation;
    ActivationFunction h_gate_activation;
    ActivationFunction r_gate_activation;
} GRUActivations;

typedef struct {
    RecurrentConfig base;
    GRUActivations activations;
} GRUConfig;

GRUActivations GRUActivationsCreate(
    ActivationFunction z_gate_activation,
    ActivationFunction h_gate_activation,
    ActivationFunction r_gate_activation
);

GRUActivations GRUActivationsCreateDefault(int size);

void GRUActivationsDestroy(GRUActivations activations);

GRUConfig GRUConfigCreate(
    int input_feature_channels,
    int output_feature_channels,
    bool return_sequences,
    int timesteps,
    GRUActivations activations
);

typedef struct GRUStruct * GRU;

GRUWeights * GRUGetWeights(GRU filter);

GRU GRUCreateForInference(GRUConfig config);

int GRUApplyInference(GRU filter, const float *input, float* output);

GRU GRUCreateForTraining(GRUConfig config, GRUTrainingConfig training_config);

GRUGradient * GRUGradientCreate(GRUConfig config, GRUTrainingConfig training_config);

int GRUApplyTrainingBatch(GRU filter, const float *input, float* output);

void GRUCalculateGradient(GRU filter, GRUGradient *gradients, float *d_out);

void GRUDestroy(GRU filter);


#if defined __cplusplus
}
#endif

#endif /* gru2_h */
