//
//  time_distributed.c
//  Pods
//
//  Created by Alex on 21.10.2020.
//

#include "nntoolkitcore/layers/time_distributed_dense.h"
#include "nntoolkitcore/core/loop.h"
#include "stdlib.h"


struct TimeDistributedDenseStruct {
    TimeDistributedDenseConfig config;
    Dense dense;
};

TimeDistributedDenseConfig TimeDistributedDenseConfigCreate(int ts, DenseConfig dense){
    TimeDistributedDenseConfig config;
    config.dense = dense;
    config.ts = ts;
    return config;
}

TimeDistributedDense TimeDistributedDenseCreate(TimeDistributedDenseConfig config){
    TimeDistributedDense ts_filter = malloc(sizeof(struct TimeDistributedDenseStruct));
    ts_filter->config = config;
    return ts_filter;
}

TimeDistributedDense TimeDistributedDenseCreateForInference(TimeDistributedDenseConfig config){
    TimeDistributedDense ts_filter = TimeDistributedDenseCreate(config);
    ts_filter->dense = DenseCreateForInference(config.dense);
    return ts_filter;
}

TimeDistributedDense TimeDistributedDenseCreateForTraining(TimeDistributedDenseConfig config, TimeDistributedDenseTrainingConfig training_config) {
    TimeDistributedDense ts_filter = TimeDistributedDenseCreate(config);
    DenseTrainingConfig dense_training_config = DefaultTrainingConfigCreate(training_config.mini_batch_size * config.ts);
    ts_filter->dense = DenseCreateForTraining(config.dense, dense_training_config);
    return ts_filter;
}

DenseWeights* TimeDistributedDenseGetWeights(TimeDistributedDense filter) {
    return DenseGetWeights(filter->dense);
}

DenseGradient* TimeDistributedDenseGradientCreate(TimeDistributedDense filter){
    return DenseGradientCreateFromFilter(filter->dense);
}

int TimeDistributedDenseApplyInference(TimeDistributedDense filter, const float *input, float* output){
    P_LOOP_START(filter->config.ts, ts)
        DenseApplyInference(filter->dense, input + ts * filter->config.dense.input_size,
                                            output + ts * filter->config.dense.output_size);
    P_LOOP_END
    return 0;
}

int TimeDistributedDenseApplyTrainingBatch(TimeDistributedDense filter, const float *input, float* output){
    return DenseApplyTrainingBatch(filter->dense, input, output);
}

void TimeDistributedDenseCalculateGradient(TimeDistributedDense filter, DenseGradient *gradient, float *d_out){
    DenseCalculateGradient(filter->dense, gradient, d_out);
}

void TimeDistributedDenseDestroy(TimeDistributedDense filter){
    DenseDestroy(filter->dense);
    free(filter);
}