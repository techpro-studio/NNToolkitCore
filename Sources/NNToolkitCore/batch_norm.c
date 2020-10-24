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

struct BatchNormFilterStruct{
    BatchNormConfig config;
    BatchNormWeights* weights;
};

BatchNormWeights* BatchNormFilterGetWeights(BatchNormFilter filter){
    return filter->weights;
}

BatchNormConfig BatchNormConfigCreate(int feature_channels, float epsilon, int batch_size){
    BatchNormConfig config;
    config.batch_size = batch_size;
    config.epsilon = epsilon;
    config.feature_channels = feature_channels;
    return config;
}

BatchNormFilter BatchNormFilterCreate(BatchNormConfig config) {
    BatchNormFilter filter = malloc(sizeof(struct BatchNormFilterStruct));
    filter->config = config;
    int chan = config.feature_channels;
    filter->weights = malloc(sizeof(BatchNormWeights));
    float *weights = malloc(4 * chan * sizeof(float));
    filter->weights->gamma = weights;
    filter->weights->beta = filter->weights->gamma + chan;
    filter->weights->mean = filter->weights->beta + chan;
    filter->weights->variance = filter->weights->mean + chan;
    return filter;
}

void BatchNormFilterDestroy(BatchNormFilter filter) {
    free(filter->weights->gamma);
    free(filter->weights);
    free(filter);
}


//output_image = (input_image - mean[c]) * gamma[c] / sqrt(variance[c] + epsilon) + beta[c];

void BatchNormFilterApplySlice(BatchNormFilter filter, const float *input, float* output){
    int size = filter->config.feature_channels;
    float buffer[size];
    // output = - mean
    op_vec_neg(filter->weights->mean, output, size);
    //  output = input + output
    op_vec_add(input, output, output, size);
    // output = (input - mean) * gamma
    op_vec_mul(output, filter->weights->gamma, output, size);
    // buffer = variance + epsilon
    op_vec_add_sc(filter->weights->variance, filter->config.epsilon, buffer, size);
//    buffer = sqrt(buffer)
    op_vec_sqrt(buffer, buffer, size);
    // output = output / buffer
    op_vec_div(output, buffer, output, size);
//    output = output + beta
    op_vec_add(output, filter->weights->beta, output, size);
}

int BatchNormFilterApply(BatchNormFilter filter, const float *input, float* output) {
    if (filter->config.batch_size == 1){
        BatchNormFilterApplySlice(filter, input, output);
        return 0;
    }

    P_LOOP_START(filter->config.batch_size, index)
        size_t offset = index * filter->config.feature_channels;
        BatchNormFilterApplySlice(filter, input + offset, output + offset);
    P_LOOP_END
    return 0;
}


