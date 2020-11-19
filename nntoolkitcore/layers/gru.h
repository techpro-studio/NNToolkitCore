//
//  gru.h
//  audio_test
//
//  Created by Alex on 21.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef gru_h
#define gru_h

#include <stdio.h>
#include "stdbool.h"
#import "activation.h"


#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    int input_feature_channels;
    int output_feature_channels;
    /*
     if true:
     h_t = (1 - z) * h_t_previous + z * h_tilda.
     else:
     h_t = (1 - z) * h_tilda + z * h_t_previous
     */
    bool flip_output_gates;
    bool v2;
    bool return_sequences;
    int timesteps;
    ActivationFunction recurrent_activation;
    ActivationFunction activation;
} GRUConfig;

GRUConfig GRUConfigCreate(
    int input_feature_channels,
    int output_feature_channels,
    bool flip_output_gates,
    bool v2,
    bool return_sequences,
    int batchSize,
    ActivationFunction recurrent_activation,
    ActivationFunction activation
);

typedef struct {
    float *W_z;
    float *U_z;

    float *b_iz;
    float *b_hz;

    float *W_r;
    float *U_r;

    float *b_ir;
    float *b_hr;

    float *W_h;
    float *U_h;
    
    float *b_ih;
    float *b_hh;

} GRUWeights;


typedef struct GRUStruct * GRU;

GRUWeights * GRUGetWeights(GRU filter);

GRU GRUCreateForInference(GRUConfig config);

int GRUApplyInference(GRU filter, const float *input, float* output);

void GRUDestroy(GRU filter);


#if defined __cplusplus
}
#endif

#endif /* gru_h */
