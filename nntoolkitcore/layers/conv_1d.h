//
//  conv_1d.h
//  mac_test
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef conv_1d_h
#define conv_1d_h

#include <stdio.h>


#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    int mini_batch_size;
} ConvTrainingConfig;


ConvTrainingConfig ConvTrainingConfigCreate(int mini_batch_size);

typedef struct {
    int input_feature_channels;
    int output_feature_channels;
    int kernel_size;
    int stride;
    int input_size;
    int output_size;
} Conv1dConfig;

Conv1dConfig Conv1dConfigCreate(int input_feature_channels, int output_feature_channels, int kernel_size, int stride, int inputSize);

typedef struct {
    float *W;
    float *b;
} ConvWeights;

typedef struct {
    float *d_W;
    float *d_b;
    float *d_X;
} ConvGradient;

typedef struct Conv1dFilterStruct* Conv1d;

ConvWeights* Conv1dGetWeights(Conv1d filter);

Conv1d Conv1dCreateForInference(Conv1dConfig config);

Conv1d Conv1dCreateForTraining(Conv1dConfig config, ConvTrainingConfig training_config);

ConvGradient* Conv1dCreateGradient(Conv1dConfig config, ConvTrainingConfig training_config);

void ConvGradientDestroy(ConvGradient *gradient);

int Conv1dApplyInference(Conv1d filter, const float *input, float* output);

int Conv1dApplyTrainingBatch(Conv1d filter, const float *input, float* output);

void Conv1dCalculateGradient(Conv1d filter, ConvGradient* gradient, const float *d_out);

void Conv1dDestroy(Conv1d filter);

#if defined __cplusplus
}
#endif


#endif /* conv_1d_h */
