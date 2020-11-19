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
#import "activation.h"


#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    ActivationFunction input_gate_activation;
    ActivationFunction update_gate_activation;
    ActivationFunction reset_gate_activation;
} GRUActivations;

typedef struct {
    int input_feature_channels;
    int output_feature_channels;
    bool return_sequences;
    int timesteps;
    GRUActivations activations;
} GRU2Config;



GRUActivations GRUActivationsCreate(
    ActivationFunction input_gate_activation,
    ActivationFunction update_gate_activation,
    ActivationFunction reset_gate_activation
);

GRUActivations GRUActivationsCreateDefault(int size);

void GRUActivationsDestroy(GRUActivations activations);

GRU2Config GRU2ConfigCreate(
    int input_feature_channels,
    int output_feature_channels,
    bool return_sequences,
    int timesteps,
    GRUActivations activations
);

typedef struct {
    float *W;
    float *U;
    float *b_i;
    float *b_h;
} GRU2Weights;


typedef struct GRU2Struct * GRU2;

GRU2Weights * GRU2GetWeights(GRU2 filter);

GRU2 GRU2CreateForInference(GRU2Config config);

//feature channels in row

int GRU2ApplyInference(GRU2 filter, const float *input, float* output);

void GRU2Destroy(GRU2 filter);


#if defined __cplusplus
}
#endif

#endif /* gru2_h */
