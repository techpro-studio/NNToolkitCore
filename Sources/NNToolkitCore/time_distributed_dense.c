//
//  time_distributed.c
//  Pods
//
//  Created by Alex on 21.10.2020.
//

#include "time_distributed_dense.h"
#include "operations.h"
#include "stdlib.h"



typedef struct  {
    TimeDistributedDenseTrainingConfig config;
    DenseTrainingConfig dense_config;
} TimeDistributedDenseTrainingData;

struct TimeDistributedDenseStruct {
    TimeDistributedDenseConfig config;
    TimeDistributedDenseTrainingData *training_data;
    DenseFilter dense;
};

TimeDistributedDenseTrainingConfig TimeDistributedDenseTrainingConfigCreate(int batch){
    TimeDistributedDenseTrainingConfig config = { batch };
    return config;
}

TimeDistributedDenseConfig TimeDistributedDenseConfigCreate(int ts, DenseConfig dense){
    TimeDistributedDenseConfig config;
    config.dense = dense;
    config.ts = ts;
    return config;
}

TimeDistributedDense TimeDistributedCreate(TimeDistributedDenseConfig config){
    TimeDistributedDense ts_filter = malloc(sizeof(struct TimeDistributedDenseStruct));
    ts_filter->config = config;
    return ts_filter;
}

TimeDistributedDense TimeDistributedCreateForInference(TimeDistributedDenseConfig config){
    TimeDistributedDense ts_filter = TimeDistributedCreate(config);
    ts_filter->dense = DenseFilterCreateForInference(config.dense);
    return ts_filter;
}

TimeDistributedDense TimeDistributedDenseCreateForTraining(TimeDistributedDenseConfig config, TimeDistributedDenseTrainingConfig training_config) {
    TimeDistributedDense ts_filter = TimeDistributedCreate(config);
    DenseTrainingConfig dense_training_config = DenseTrainingConfigCreate(training_config.mini_batch_size * config.ts);
    ts_filter->dense = DenseFilterCreateForTraining(config.dense, dense_training_config);
    ts_filter->training_data = malloc(sizeof(TimeDistributedDenseTrainingData));
    ts_filter->training_data->config = training_config;
    ts_filter->training_data->dense_config = dense_training_config;
    return ts_filter;
}

int TimeDistributedDenseFilterApply(TimeDistributedDense filter, const float *input, float* output){
    P_LOOP_START(filter->config.ts, ts)
        DenseFilterApply(filter->dense, input + ts * filter->config.dense.inputSize
                         , output + ts * filter->config.dense.outputSize);
    P_LOOP_END
    return 0;
}

int TimeDistributedDenseFilterApplyTrainingBatch(TimeDistributedDense filter, const float *input, float* output){
    if (filter->training_data == NULL){
        return -1;
    }
    DenseFilterApplyTrainingBatch(filter->dense, input, output);
    return 0;
}

void TimeDistributedDenseFilterCalculateGradient(TimeDistributedDense filter, DenseGradient *gradient, float *d_out){
    DenseGradient *dense_gradient = DenseGradientCreate(filter->config.dense, filter->training_data->dense_config);
    DenseFilterCalculateGradient(filter->dense, dense_gradient, d_out);
    int d_W_size = filter->config.dense.inputSize * filter->config.dense.outputSize;
    int d_b_size = filter->config.dense.outputSize;
    int ts = filter->config.ts;
    int batch = filter->training_data->config.mini_batch_size;
    P_LOOP_START(batch, b)
        for (int t = 0; t < ts; ++t){
            VectorAdd(gradient->d_W, dense_gradient->d_W + (t * d_W_size + b * ts * d_W_size), gradient->d_W, d_W_size);
            VectorAdd(gradient->d_b, dense_gradient->d_b + (t * d_b_size + b * ts * d_b_size), gradient->d_b, d_b_size);
        }
    P_LOOP_END
    memcpy(gradient->d_X, dense_gradient->d_X, filter->config.dense.inputSize * ts * batch * sizeof(float));
}

void TimeDistributedDenseDestroy(TimeDistributedDense filter){
    DenseFilterDestroy(filter->dense);
    free(filter);
}
