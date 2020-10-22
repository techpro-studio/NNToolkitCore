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

typedef struct TimeDistributedDenseStruct* TimeDistributedDense;

TimeDistributedDense TimeDistributedDenseCreate(int ts, DenseFilter filter);

int TimeDistributedDenseFilterApply(TimeDistributedDense filter, const float *input, float* output);

int TimeDistributedDenseFilterApplyTrainingBatch(TimeDistributedDense filter, const float *input, float* output);

void TimeDistributedDestroy(TimeDistributedDense filter);

#endif /* time_distributed_h */
