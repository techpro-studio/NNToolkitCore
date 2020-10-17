//
//  batch_norm.c
//  audio_test
//
//  Created by Alex on 24.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "batch_norm.h"
#import "operations.h"
#include "stdlib.h"
#include <dispatch/dispatch.h>

struct BatchNormFilterStruct{
    BatchNormConfig config;
    BatchNormWeights* weights;
};

BatchNormWeights* BatchNormFilterGetWeights(BatchNormFilter filter){
    return filter->weights;
}

BatchNormConfig BatchNormConfigCreate(int featureChannels, float epsilon, int batchSize){
    BatchNormConfig config;
    config.batchSize = batchSize;
    config.epsilon = epsilon;
    config.featureChannels = featureChannels;
    return config;
}

BatchNormFilter BatchNormFilterCreate(BatchNormConfig config) {
    BatchNormFilter filter = malloc(sizeof(struct BatchNormFilterStruct));
    filter->config = config;
    filter->weights = malloc(sizeof(BatchNormWeights));
    float *weights = malloc(4 * config.featureChannels * sizeof(float));
    filter->weights->gamma = weights;
    filter->weights->beta = filter->weights->gamma + config.featureChannels;
    filter->weights->mean = filter->weights->beta + config.featureChannels;
    filter->weights->variance = filter->weights->mean + config.featureChannels;
    return filter;
}

void BatchNormFilterDestroy(BatchNormFilter filter) {
    free(filter->weights->gamma);
    free(filter->weights);
    free(filter);
}


//output_image = (input_image - mean[c]) * gamma[c] / sqrt(variance[c] + epsilon) + beta[c];

void BatchNormFilterApplySlice(BatchNormFilter filter, const float *input, float* output){
    int size = filter->config.featureChannels;
    float buffer[size];
    // output = - mean
    VectorNeg(filter->weights->mean, output, size);
    //  output = input + output
    VectorAdd(input, output, output, size);
    // output = (input - mean) * gamma
    VectorMul(output, filter->weights->gamma,output, size);
    // buffer = variance + epsilon
    VectorAddS(filter->weights->variance, filter->config.epsilon, buffer, size);
//    buffer = sqrt(buffer)
    VectorSqrt(buffer, buffer, size);
    // output = output / buffer
    VectorDiv(output, buffer, output, size);
//    output = output + beta
    VectorAdd(output, filter->weights->beta, output, size);
}

void BatchNormFilterApply(BatchNormFilter filter, const float *input, float* output) {
    if (filter->config.batchSize == 1){
        BatchNormFilterApplySlice(filter, input, output);
        return;
    }
    dispatch_apply(filter->config.batchSize, DISPATCH_APPLY_AUTO, ^(size_t index) {
        size_t offset = index * filter->config.featureChannels;
        BatchNormFilterApplySlice(filter, input + offset, output + offset);
    });
}


