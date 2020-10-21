//
//  time_distributed.h
//  Pods
//
//  Created by Alex on 21.10.2020.
//

#ifndef time_distributed_h
#define time_distributed_h

#include <stdio.h>

typedef struct {
    int ts;
    int input_ts_offset;
    int output_ts_offset;
} TimeDistributedConfig;


typedef int (*filter_apply_fn)(void *filter, const float *input, float* output);
typedef int (*filter_calculate_gradient_fn)(void *filter, void* gradients, float* dout);

TimeDistributedConfig TimeDistributedConfigCreate(int ts, int input_ts_offset, int output_ts_offset);

typedef struct TimeDistributedStruct* TimeDistributed;

TimeDistributed TimeDistributedCreate(TimeDistributedConfig config, void *filter, filter_apply_fn apply_fn, filter_calculate_gradient_fn back_fn);

int TimeDistributedFilterApply(TimeDistributed filter, const float *input, float* output);

void TimeDistributedDestroy(TimeDistributed filter);

#endif /* time_distributed_h */
