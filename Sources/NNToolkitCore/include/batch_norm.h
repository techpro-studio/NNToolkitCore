//
//  batch_norm.h
//  audio_test
//
//  Created by Alex on 24.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef batch_norm_h
#define batch_norm_h

#include <stdio.h>


#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    float *gamma;
    float *beta;
    float *mean;
    float *variance;
} BatchNormWeights;


typedef struct {
    int feature_channels;
    float epsilon;
    int batch_size;
} BatchNormConfig;

typedef struct BatchNormFilterStruct* BatchNormFilter;

BatchNormWeights* BatchNormFilterGetWeights(BatchNormFilter filter);

BatchNormConfig BatchNormConfigCreate(int featureChannels, float epsilon, int batchSize);

BatchNormFilter BatchNormFilterCreate(BatchNormConfig config);

int BatchNormFilterApply(BatchNormFilter filter, const float *input, float* output);

void BatchNormFilterDestroy(BatchNormFilter filter);


#if defined __cplusplus
}
#endif




#endif /* batch_norm_h */
