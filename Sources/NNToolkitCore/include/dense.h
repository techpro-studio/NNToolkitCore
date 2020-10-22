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
    int inputSize;
    int outputSize;
    ActivationFunction activation;
} DenseConfig;

typedef struct DenseFilterStruct* DenseFilter;

DenseWeights* DenseFilterGetWeights(DenseFilter filter);

DenseConfig DenseConfigCreate(int inputSize, int outputSize, ActivationFunction activation);

DenseFilter DenseFilterCreate(DenseConfig config);

DenseFilter DenseFilterCreateForTraining(DenseConfig config);

int DenseFilterApply(DenseFilter filter, const float *input, float* output);

void DenseFilterDestroy(DenseFilter filter);


#if defined __cplusplus
}
#endif

#endif /* dense_h */
