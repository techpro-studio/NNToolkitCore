//
//  time_distributed.c
//  Pods
//
//  Created by Alex on 21.10.2020.
//

#include "time_distributed_dense.h"
#include "operations.h"
#include "stdlib.h"

struct TimeDistributedDenseStruct {
    int ts;
    DenseFilter dense;
};


TimeDistributedDense TimeDistributedCreate(int ts, DenseFilter dense){
    TimeDistributedDense ts_filter = malloc(sizeof(struct TimeDistributedDenseStruct));
    ts_filter->ts = ts;
    ts_filter->dense = dense;
    return ts_filter;
}

int TimeDistributedDenseFilterApply(TimeDistributedDense filter, const float *input, float* output){
    DenseConfig config = DenseFilterGetConfig(filter->dense);
    P_LOOP_START(filter->ts, ts)
        DenseFilterApply(filter->dense, input + ts * config.inputSize
                         , output + ts * config.outputSize);
    P_LOOP_END
    return 0;
}

void TimeDistributedDenseDestroy(TimeDistributedDense filter){
    free(filter);
}
