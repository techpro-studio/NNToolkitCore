//
// Created by Alex on 20.11.2020.
//

#ifndef recurrent_h
#define recurrent_h

#include "shared.h"
#include "stdbool.h"


#if defined __cplusplus
extern "C" {
#endif


typedef struct {
    int w;
    int u;
    int b_i;
    int b_h;
    int sum;
} RecurrentWeightsSize;

typedef struct {
    float *W;
    float *U;
    float *b_i;
    float *b_h;
} RecurrentWeights;

typedef DefaultTrainingConfig RecurrentTrainingConfig;

typedef struct {
    int input_feature_channels;
    int output_feature_channels;
    bool return_sequences;
    int timesteps;
} RecurrentConfig;

typedef struct {
    float *d_W;
    float *d_U;
    float *d_b_i;
    float *d_b_h;
    float *d_X;
} RecurrentGradient;



inline static RecurrentConfig RecurrentConfigCreate(
        int input_feature_channels,
        int output_feature_channels,
        bool return_sequences,
        int timesteps
) {
    return (RecurrentConfig) {
            .input_feature_channels = input_feature_channels,
            .timesteps = timesteps,
            .return_sequences = return_sequences,
            .output_feature_channels = output_feature_channels
    };
}

#if defined __cplusplus
}
#endif

#endif //recurrent
