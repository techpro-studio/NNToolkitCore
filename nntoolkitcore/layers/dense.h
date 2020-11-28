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
#include "shared.h"


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

typedef DefaultTrainingConfig DenseTrainingConfig;

typedef struct {
    int input_size;
    int output_size;
    ActivationFunction activation;
} DenseConfig;

DenseGradient* DenseGradientCreate(DenseConfig config, DenseTrainingConfig training_config);

void DenseGradientDestroy(DenseGradient *gradient);

typedef struct DenseStruct* Dense;

DenseWeights* DenseGetWeights(Dense filter);

DenseConfig DenseConfigCreate(int input_size, int output_size, ActivationFunction activation);

Dense DenseCreateForInference(DenseConfig config);

Dense DenseCreateForTraining(DenseConfig config, DenseTrainingConfig training_config);

int DenseApplyInference(Dense filter, const float *input, float* output);

int DenseApplyTrainingBatch(Dense filter, const float *input, float* output);

void DenseCalculateGradient(Dense filter, DenseGradient *gradient, float *d_out);

void DenseDestroy(Dense filter);

#if defined __cplusplus
}
#endif

#endif /* dense_h */
