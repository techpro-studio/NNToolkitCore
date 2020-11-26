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
#include "shared.h"

#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    DenseConfig dense;
    int ts;
} TimeDistributedDenseConfig;

typedef DefaultTrainingConfig TimeDistributedDenseTrainingConfig;

TimeDistributedDenseConfig TimeDistributedDenseConfigCreate(int ts, DenseConfig dense);

typedef struct TimeDistributedDenseStruct *TimeDistributedDense;

DenseWeights *TimeDistributedDenseGetWeights(TimeDistributedDense filter);

DenseGradient *TimeDistributedDenseGradientCreate(TimeDistributedDense filter);

TimeDistributedDense TimeDistributedDenseCreateForInference(TimeDistributedDenseConfig config);

TimeDistributedDense TimeDistributedDenseCreateForTraining(TimeDistributedDenseConfig config,
                                                           TimeDistributedDenseTrainingConfig training_config);

int TimeDistributedDenseApplyInference(TimeDistributedDense filter, const float *input, float *output);

int TimeDistributedDenseApplyTrainingBatch(TimeDistributedDense filter, const float *input, float *output);

void TimeDistributedDenseCalculateGradient(TimeDistributedDense filter, DenseGradient *gradient, float *d_out);

void TimeDistributedDenseDestroy(TimeDistributedDense filter);

#if defined __cplusplus
}
#endif

#endif /* time_distributed_h */
