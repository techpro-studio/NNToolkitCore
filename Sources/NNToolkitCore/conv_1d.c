//
//  conv_1d.c
//  mac_test
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "conv_1d.h"
#include "stdlib.h"
#include <dispatch/dispatch.h>
#include "operations.h"

struct Conv1dFilterStruct {
    Conv1dConfig config;
    Conv1dWeights* weights;
    void *buffer;
    void *v_dot;
} ;

Conv1dWeights* Conv1dFilterGetWeights(Conv1dFilter filter){
    return filter->weights;
}

Conv1dConfig Conv1dConfigCreate(int inputFeatureChannels, int outputFeatureChannels, int kernelSize, int stride, int inputSize){
    Conv1dConfig config;
    config.inputFeatureChannels = inputFeatureChannels;
    config.outputFeatureChannels = outputFeatureChannels;
    config.inputSize = inputSize;
    config.kernelSize = kernelSize;
    config.stride = stride;
    config.outputSize = ((inputSize - (kernelSize - stride)) / stride);
    return config;
}


Conv1dFilter Conv1dFilterCreate(Conv1dConfig config) {
    Conv1dFilter filter = malloc(sizeof(struct Conv1dFilterStruct));
    filter->config = config;
    //w8s
    filter->weights = malloc(sizeof(Conv1dWeights));
    filter->weights->W = malloc(config.kernelSize * config.inputFeatureChannels * config.outputFeatureChannels * sizeof(float));
    filter->weights->b = malloc(config.outputFeatureChannels * sizeof(float));
    filter->v_dot = GetOptimized(config.kernelSize);
    filter->buffer = malloc(config.inputSize * config.inputFeatureChannels * sizeof(float));
    return filter;
}

void Conv1dFilterDestroy(Conv1dFilter filter){
    free(filter->buffer);
    free(filter->weights->W);
    free(filter->weights->b);
    free(filter->weights);
    free(filter);
}

void Conv1dFilterApply(Conv1dFilter filter, const float *input, float* output){
    float *floatInput = (float*)filter->buffer;
    MatTrans((float *) input, floatInput, filter->config.inputFeatureChannels, filter->config.inputSize);
    int kernelSize = filter->config.kernelSize;
    VectorDotF fn = (VectorDotF) filter->v_dot;
    dispatch_apply(filter->config.outputFeatureChannels, DISPATCH_APPLY_AUTO, ^(size_t outFeature) {
        for (int x = 0; x < filter->config.outputSize; ++x){
            int weightsOffset = (int)outFeature * filter->config.inputFeatureChannels * kernelSize;
            const float *outputFeatureWeights = filter->weights->W + weightsOffset;

            float result = 0.0f;

            int inputRowOffset = x * filter->config.stride;

            for (int i = 0; i < filter->config.inputFeatureChannels; ++i)
            {
                const float* rowPtr = floatInput + i * filter->config.inputSize + inputRowOffset;
                const float* weightsPtr = outputFeatureWeights + (i * kernelSize);
                result += fn(rowPtr, weightsPtr, kernelSize);
            }

            result += filter->weights->b[outFeature];

            ((float *)output)[x * filter->config.outputFeatureChannels + outFeature] = result;
        }
    });
}


















