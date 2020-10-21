//
//  time_distributed.c
//  Pods
//
//  Created by Alex on 21.10.2020.
//

#include "time_distributed.h"
#include "operations.h"
#include "stdlib.h"

struct TimeDistributedStruct{
    TimeDistributedConfig config;
    void *filter;
    filter_apply_fn apply_fn;
    filter_calculate_gradient_fn back_fn;
};

TimeDistributedConfig TimeDistributedConfigCreate(int ts, int input_ts_offset, int output_ts_offset){
    TimeDistributedConfig config;
    config.ts = ts;
    config.input_ts_offset = input_ts_offset;
    config.output_ts_offset = output_ts_offset;
    return config;
}

TimeDistributed TimeDistributedCreate(TimeDistributedConfig config, void *filter, filter_apply_fn apply_fn, filter_calculate_gradient_fn back_fn){
    TimeDistributed ts_filter = malloc(sizeof(struct TimeDistributedStruct));
    ts_filter->config = config;
    ts_filter->filter = filter;
    ts_filter->apply_fn = apply_fn;
    ts_filter->back_fn = back_fn;
    return ts_filter;
}

int TimeDistributedFilterApply(TimeDistributed filter, const float *input, float* output){
    P_LOOP_START(filter->config.ts, ts)
        filter->apply_fn(filter->filter, input + ts * filter->config.input_ts_offset
                         , output + ts * filter->config.output_ts_offset);
    P_LOOP_END
    return 0;
}

void TimeDistributedDestroy(TimeDistributed filter){
    free(filter);
}
