//
//  conv_1d.c
//  mac_test
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "conv_1d.h"
#include "stdlib.h"
#include "operations.h"

struct Conv1dFilterStruct {
    Conv1dConfig config;
    Conv1dWeights* weights;
    void *buffer;
    void *v_dot;
} ;

Conv1dWeights* Conv1dGetWeights(Conv1d filter){
    return filter->weights;
}

Conv1dConfig Conv1dConfigCreate(int input_feature_channels, int output_feature_channels, int kernel_size, int stride, int inputSize){
    Conv1dConfig config;
    config.input_feature_channels = input_feature_channels;
    config.output_feature_channels = output_feature_channels;
    config.input_size = inputSize;
    config.kernel_size = kernel_size;
    config.stride = stride;
    config.output_size = ((inputSize - (kernel_size - stride)) / stride);
    return config;
}


Conv1d Conv1dCreateForInference(Conv1dConfig config) {
    Conv1d filter = malloc(sizeof(struct Conv1dFilterStruct));
    filter->config = config;
    //w8s
    filter->weights = malloc(sizeof(Conv1dWeights));
    filter->weights->W = malloc(config.kernel_size * config.input_feature_channels * config.output_feature_channels * sizeof(float));
    filter->weights->b = malloc(config.output_feature_channels * sizeof(float));
    filter->v_dot = op_vec_dot_get_optimized(config.kernel_size);
    filter->buffer = malloc(config.input_size * config.input_feature_channels * sizeof(float));
    return filter;
}

void Conv1dDestroy(Conv1d filter){
    free(filter->buffer);
    free(filter->weights->W);
    free(filter->weights->b);
    free(filter->weights);
    free(filter);
}

int Conv1dApplyInference(Conv1d filter, const float *input, float* output){
//    if(filter->training_data != NULL){
//        return -1;
//    }
    float *floatInput = (float*)filter->buffer;
    op_mat_transp((float *) input, floatInput, filter->config.input_feature_channels, filter->config.input_size);
    int kernelSize = filter->config.kernel_size;
    op_vec_dot_fn fn = (op_vec_dot_fn) filter->v_dot;
    P_LOOP_START(filter->config.output_feature_channels, outFeature)
        for (int x = 0; x < filter->config.output_size; ++x){
            int weightsOffset = (int)outFeature * filter->config.input_feature_channels * kernelSize;
            const float *outputFeatureWeights = filter->weights->W + weightsOffset;

            float result = 0.0f;

            int inputRowOffset = x * filter->config.stride;

            for (int i = 0; i < filter->config.input_feature_channels; ++i)
            {
                const float* rowPtr = floatInput + i * filter->config.input_size + inputRowOffset;
                const float* weightsPtr = outputFeatureWeights + (i * kernelSize);
                result += fn(rowPtr, weightsPtr, kernelSize);
            }

            result += filter->weights->b[outFeature];

            ((float *)output)[x * filter->config.output_feature_channels + outFeature] = result;
        }
    P_LOOP_END
    return 0;
}


















