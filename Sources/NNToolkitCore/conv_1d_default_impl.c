//
//  conv_1d_defaut_impl.c
//  audio_test
//
//  Created by Alex on 27.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "conv_1d_default_impl.h"
#include "stdlib.h"
#include <simd/simd.h>
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>


typedef float(*compute_conv1d_kernel_f)(const float* rowPtr, const float *weightsPtr, int kernelSize);

#define compute_conv1d_kernel(NUM)  float compute_conv1d_kernel_##NUM(const float* rowPtr, const float *weightsPtr, int kernelSize)\
{\
    float sum = 0.0f;\
    int iterations = kernelSize / NUM;\
    for (int i = 0; i < iterations; ++i)\
    {\
        simd_float##NUM row = ((simd_float##NUM*) rowPtr)[i];\
        simd_float##NUM weights = ((simd_float##NUM*) weightsPtr)[i];\
        sum += simd_dot(row, weights);\
    }\
    int left = kernelSize % NUM;\
    for (int i = 0; i < left; ++i)\
    {\
        sum += rowPtr[iterations * NUM + i] * weightsPtr[iterations * NUM + i];\
    }\
    return sum;\
}

compute_conv1d_kernel(2)
compute_conv1d_kernel(3)
compute_conv1d_kernel(4)
compute_conv1d_kernel(8)
compute_conv1d_kernel(16)



typedef struct{
    compute_conv1d_kernel_f kernelFn;
    float* transposeBuffer;
    Conv1dFilter* filter;
} Conv1dDefaultImplementer;



typedef enum {
    two = 2, three = 3, four = 4, eight = 8, sixteen = 16
}optimal_vector_size;

optimal_vector_size getOptimalVectorSize(int size){
    int optimalIndex = 0;
    int smallestValue = 230;
    optimal_vector_size values [] = {two, three, four, eight, sixteen};
    int full[5] = {size / 2, size / 3, size / 4, size / 8, size / 16};
    int last[5] = {size % 2, size % 3, size % 4, size % 8, size % 16};
    for (int i = 0; i < 5; ++i){
        if (full[i] == 0){
            continue;
        }
        int sum = full[i] + last[i];
        if (sum < smallestValue){
            smallestValue = sum;
            optimalIndex = i;
        }
    }
    return values[optimalIndex];
}

Conv1dDefaultImplementer* Conv1dDefautImplementerCreate(Conv1dFilter* filter) {
    Conv1dDefaultImplementer* implementer = malloc(sizeof(Conv1dDefaultImplementer));
    implementer->transposeBuffer = malloc(filter->config.inputSize * filter->config.inputFeatureChannels * sizeof(float));
    implementer->filter = filter;
    optimal_vector_size value = getOptimalVectorSize(filter->config.kernelSize);
    switch (value) {
        case two:
            implementer->kernelFn = compute_conv1d_kernel_2;
            break;
        case three:
            implementer->kernelFn = compute_conv1d_kernel_3;
            break;
        case four:
            implementer->kernelFn = compute_conv1d_kernel_4;
            break;
        case eight:
            implementer->kernelFn = compute_conv1d_kernel_8;
            break;
        case sixteen:
            implementer->kernelFn = compute_conv1d_kernel_16;
            break;
    }
    return implementer;
}


int Conv1dDefaultImplementerApply(void *implementer, const void *input, void *output){
    Conv1dDefaultImplementer impl = *((Conv1dDefaultImplementer *) implementer);
    float * floatInput = impl.transposeBuffer;
    Conv1dFilter *filter = impl.filter;
    compute_conv1d_kernel_f computeKernel = impl.kernelFn;
    vDSP_mtrans((float *) input, 1, floatInput, 1, filter->config.inputFeatureChannels, filter->config.inputSize);
    
    int kernelSize = filter->config.kernelSize;
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
                result += computeKernel(rowPtr, weightsPtr, kernelSize);
            }

            result += filter->weights->b[outFeature];

            ((float *)output)[x * filter->config.outputFeatureChannels + outFeature] = result;
        }
    });

    return 0;
}

void Conv1dDefautImplementerDestroy(void *implementer){
    Conv1dDefaultImplementer *defaultImplementer = (Conv1dDefaultImplementer *) implementer;
    defaultImplementer->filter = NULL;
    free(defaultImplementer->transposeBuffer);
    free(defaultImplementer);
}

void Conv1dFilterSetupDefaultImplementer(Conv1dFilter *filter){
    Conv1dDefaultImplementer* defaultImpementer = Conv1dDefautImplementerCreate(filter);
    filter->implementer = Conv1dImplementerCreate(Conv1dDefaultImplementerApply, Conv1dDefautImplementerDestroy, defaultImpementer);
}
