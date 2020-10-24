//
//  time_distributed.h
//  Pods
//
//  Created by Alex on 21.10.2020.
//

#ifndef time_distributed_h
#define time_distributed_h

#include <stdio.h>
#include "dense.h"

typedef struct {
    DenseConfig dense;
    int ts;
} TimeDistributedDenseConfig;

typedef struct {
    int mini_batch_size;
} TimeDistributedDenseTrainingConfig;

TimeDistributedDenseConfig TimeDistributedDenseConfigCreate(int ts, DenseConfig dense);

TimeDistributedDenseTrainingConfig TimeDistributedDenseTrainingConfigCreate(int batch);

typedef struct TimeDistributedDenseStruct* TimeDistributedDense;

DenseWeights* TimeDistributedDenseFilterGetWeights(TimeDistributedDense filter);

DenseGradient* TimeDistributedDenseGradientCreate(TimeDistributedDense filter);

TimeDistributedDense TimeDistributedDenseCreateForInference(TimeDistributedDenseConfig config);

TimeDistributedDense TimeDistributedDenseCreateForTraining(TimeDistributedDenseConfig config, TimeDistributedDenseTrainingConfig training_config);

int TimeDistributedDenseFilterApply(TimeDistributedDense filter, const float *input, float* output);

int TimeDistributedDenseFilterApplyTrainingBatch(TimeDistributedDense filter, const float *input, float* output);

void TimeDistributedDenseFilterCalculateGradient(TimeDistributedDense filter, DenseGradient *gradient, float *d_out);

void TimeDistributedDenseDestroy(TimeDistributedDense filter);

#endif /* time_distributed_h */
