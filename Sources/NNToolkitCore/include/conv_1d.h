//
//  conv_1d.h
//  mac_test
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef conv_1d_h
#define conv_1d_h

#include <stdio.h>


#if defined __cplusplus
extern "C" {
#endif



typedef struct {
    int inputFeatureChannels;
    int outputFeatureChannels;
    int kernelSize;
    int stride;
    int inputSize;
    int outputSize;
} Conv1dConfig;

Conv1dConfig Conv1dConfigCreate(int inputFeatureChannels, int outputFeatureChannels, int kernelSize, int stride, int inputSize);

typedef struct {
    float *W;
    float *b;
} Conv1dWeights;


typedef struct {
    Conv1dConfig config;
    Conv1dWeights* weights;
    void *buffer;
    void *v_dot;
} Conv1dFilter;

Conv1dFilter* Conv1dFilterCreate(Conv1dConfig config);

void Conv1dFilterApply(Conv1dFilter *filter, const float *input, float* output);

void Conv1dFilterDestroy(Conv1dFilter *filter);

#if defined __cplusplus
}
#endif


#endif /* conv_1d_h */
