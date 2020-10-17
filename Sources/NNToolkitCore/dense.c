//
//  dense.c
//  audio_test
//
//  Created by Alex on 29.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "dense.h"
#include "operations.h"
#include "stdlib.h"
#include <dispatch/dispatch.h>

struct DenseFilterStruct {
    DenseConfig config;
    DenseWeights* weights;
};

DenseWeights* DenseFilterGetWeights(DenseFilter filter){
    return filter->weights;
}

DenseConfig DenseConfigCreate(int inputSize, int outputSize, ActivationFunction activation){
    DenseConfig config;
    config.inputSize = inputSize;
    config.outputSize = outputSize;
    config.activation = activation;
    return config;
}

DenseFilter DenseFilterCreate(DenseConfig config) {
    DenseFilter filter = malloc(sizeof(struct DenseFilterStruct));
    filter->config = config;
    filter->weights = malloc(sizeof(DenseWeights));
    filter->weights->W = malloc(config.inputSize * (config.outputSize + 1) * sizeof(float));
    filter->weights->b = filter->weights->W + config.inputSize * config.outputSize;
    return filter;
}

void DenseFilterDestroy(DenseFilter filter) {
    free(filter->weights->W);
    free(filter->weights);
    free(filter);
}

void DenseFilterApply(DenseFilter filter, const float *input, float* output) {
    MatMul(input, filter->weights->W, output, 1,  filter->config.outputSize, filter->config.inputSize, 0.0);
    VectorAdd(output, filter->weights->b, output, filter->config.outputSize);
    if (filter->config.activation) {
        ActivationFunctionApply(filter->config.activation, output, output);
    }
}


void DenseFilterApplyTimeDistributed(DenseFilter filter, int size, const float *input, float* output) {
    dispatch_apply(size, DISPATCH_APPLY_AUTO, ^(size_t i) {
        DenseFilterApply(filter, input + i * filter->config.inputSize, output + i * filter->config.outputSize);
    });

}

