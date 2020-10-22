//
//  dense.h
//  audio_test
//
//  Created by Alex on 29.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef dense_h
#define dense_h

#include <stdio.h>
#include "activation.h"


#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    float *W;
    float *b;
} DenseWeights;

typedef struct {
    float *d_W;
    float *d_b;
    float *d_X;
} DenseGradient;

typedef struct {
    int mini_batch_size;
} DenseTrainingConfig;

DenseTrainingConfig DenseTrainingConfigCreate(int batch);

typedef struct {
    int inputSize;
    int outputSize;
    ActivationFunction activation;
} DenseConfig;

DenseGradient* DenseGradientCreate(DenseConfig config, DenseTrainingConfig trainingConfig);

void DenseGradientDestroy(DenseGradient *gradient);

typedef struct DenseFilterStruct* DenseFilter;

DenseWeights* DenseFilterGetWeights(DenseFilter filter);

DenseConfig DenseConfigCreate(int inputSize, int outputSize, ActivationFunction activation);

DenseFilter DenseFilterCreateForInference(DenseConfig config);

DenseFilter DenseFilterCreateForTraining(DenseConfig config, DenseTrainingConfig training_config);

DenseConfig DenseFilterGetConfig(DenseFilter filter);

int DenseFilterApply(DenseFilter filter, const float *input, float* output);

int DenseFilterApplyTrainingBatch(DenseFilter filter, const float *input, float* output);

void DenseFilterCalculateGradient(DenseFilter filter, DenseGradient *gradient, float *d_out);

void DenseFilterDestroy(DenseFilter filter);

#if defined __cplusplus
}
#endif

#endif /* dense_h */
