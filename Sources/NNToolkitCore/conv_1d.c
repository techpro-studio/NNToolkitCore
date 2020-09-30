//
//  conv_1d.c
//  mac_test
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "conv_1d.h"
#include "stdlib.h"
#include <simd/simd.h>


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

Conv1dFilter* Conv1dFilterCreate(Conv1dConfig config) {

    Conv1dFilter* filter = malloc(sizeof(Conv1dFilter));
    filter->config = config;
    //w8s
    filter->weights = malloc(sizeof(Conv1dWeights));
    filter->weights->W = malloc(config.kernelSize * config.inputFeatureChannels * config.outputFeatureChannels * sizeof(float));
    filter->weights->b = malloc(config.outputFeatureChannels * sizeof(float));

    return filter;
}

void Conv1dFilterDestroy(Conv1dFilter *filter){
    if (filter->implementer){
        filter->implementer->destroyFn(filter->implementer->ptr);
        free(filter->implementer);
    }
    free(filter->weights->W);
    free(filter->weights->b);
    free(filter->weights);
    free(filter);
}

Conv1dImplementer* Conv1dImplementerCreate(ImplemnterApply applyFn, ImplemnterDestroy destroyFn, void *ptr){
    Conv1dImplementer* implementer = malloc(sizeof(Conv1dImplementer));
    implementer->destroyFn = destroyFn;
    implementer->ptr = ptr;
    implementer->applyFn = applyFn;
    return implementer;
}

void Conv1dFilterApply(Conv1dFilter *filter, const float *input, float* output){
    filter->implementer->applyFn(filter->implementer->ptr, input, output);
}



















