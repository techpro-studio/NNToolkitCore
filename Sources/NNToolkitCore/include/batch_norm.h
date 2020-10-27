//
//  batch_norm.h
//  audio_test
//
//  Created by Alex on 24.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef batch_norm_h
#define batch_norm_h

#include <stdio.h>


#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    float *gamma;
    float *beta;
    float *mean;
    float *variance;
} BatchNormWeights;


typedef struct {
    float momentum;
    int mini_batch_size;
} BatchNormTrainingConfig;

typedef struct {
    float *d_gamma;
    float *d_beta;
} BatchNormGradient;

typedef struct {
    int feature_channels;
    float epsilon;
    int count;
} BatchNormConfig;

typedef struct BatchNormFilterStruct* BatchNorm;

BatchNormWeights* BatchNormGetWeights(BatchNorm filter);

BatchNormConfig BatchNormConfigCreate(int feature_channels, float epsilon, int count);

BatchNormTrainingConfig BatchNormTrainingConfigCreate(float momentum, int mini_batch_size);

BatchNorm BatchNormCreateForInference(BatchNormConfig config);

BatchNorm BatchNormCreateForTraining(BatchNormConfig config, BatchNormTrainingConfig training_config);

BatchNormGradient* BatchNormGradientCreate(BatchNormConfig config, BatchNormTrainingConfig training_config);

int BatchNormApplyInference(BatchNorm filter, const float *input, float* output);

int BatchNormApplyTrainingBatch(BatchNorm filter, const float *input, float* output);

void BatchNormDestroy(BatchNorm filter);


#if defined __cplusplus
}
#endif




#endif /* batch_norm_h */
