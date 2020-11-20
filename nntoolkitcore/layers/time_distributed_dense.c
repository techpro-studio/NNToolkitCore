//
//  time_distributed.c
//  Pods
//
//  Created by Alex on 21.10.2020.
//

#include <nntoolkitcore/core/memory.h>
#include "nntoolkitcore/layers/time_distributed_dense.h"
#include "nntoolkitcore/core/ops.h"
#include "nntoolkitcore/core/loop.h"
#include "stdlib.h"
#include "string.h"


typedef struct  {
    TimeDistributedDenseTrainingConfig config;
    DenseTrainingConfig dense_config;
} TimeDistributedDenseTrainingData;

struct TimeDistributedDenseStruct {
    TimeDistributedDenseConfig config;
    TimeDistributedDenseTrainingData *training_data;
    Dense dense;
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

TimeDistributedDense TimeDistributedDenseCreate(TimeDistributedDenseConfig config){
    TimeDistributedDense ts_filter = malloc(sizeof(struct TimeDistributedDenseStruct));
    ts_filter->config = config;
    return ts_filter;
}

TimeDistributedDense TimeDistributedDenseCreateForInference(TimeDistributedDenseConfig config){
    TimeDistributedDense ts_filter = TimeDistributedDenseCreate(config);
    ts_filter->dense = DenseCreateForInference(config.dense);
    ts_filter->training_data = NULL;
    return ts_filter;
}


TimeDistributedDense TimeDistributedDenseCreateForTraining(TimeDistributedDenseConfig config, TimeDistributedDenseTrainingConfig training_config) {
    TimeDistributedDense ts_filter = TimeDistributedDenseCreate(config);
    DenseTrainingConfig dense_training_config = DenseTrainingConfigCreate(training_config.mini_batch_size * config.ts);
    ts_filter->dense = DenseCreateForTraining(config.dense, dense_training_config);
    ts_filter->training_data = malloc(sizeof(TimeDistributedDenseTrainingData));
    ts_filter->training_data->config = training_config;
    ts_filter->training_data->dense_config = dense_training_config;
    return ts_filter;
}

DenseWeights* TimeDistributedDenseGetWeights(TimeDistributedDense filter) {
    return DenseGetWeights(filter->dense);
}

DenseGradient* TimeDistributedDenseGradientCreate(TimeDistributedDense filter){
    DenseGradient* grad = malloc(sizeof(DenseGradient));

    int in = filter->config.dense.input_size;
    int out = filter->config.dense.output_size;
    int batch = filter->training_data->config.mini_batch_size;
    int ts = filter->config.ts;

    int d_w_size =  in * out * ts * batch;
    int d_x_size = in * ts * batch;
    int buff_size = (d_w_size + d_x_size + out * ts * batch) * sizeof(float);
    grad->d_W = (float *) malloc_zeros(buff_size);
    grad->d_X = grad->d_W + d_w_size;
    grad->d_b = grad->d_X + d_x_size;
    return grad;
}


int TimeDistributedDenseApplyInference(TimeDistributedDense filter, const float *input, float* output){
    if(filter->training_data != NULL){
        return -1;
    }
    P_LOOP_START(filter->config.ts, ts)
                        DenseApplyInference(filter->dense, input + ts * filter->config.dense.input_size,
                                            output + ts * filter->config.dense.output_size);
    P_LOOP_END
    return 0;
}

int TimeDistributedDenseApplyTrainingBatch(TimeDistributedDense filter, const float *input, float* output){
    if (filter->training_data == NULL){
        return -1;
    }
    DenseApplyTrainingBatch(filter->dense, input, output);
    return 0;
}

void TimeDistributedDenseCalculateGradient(TimeDistributedDense filter, DenseGradient *gradient, float *d_out){
    DenseGradient *dense_gradient = DenseGradientCreate(filter->config.dense, filter->training_data->dense_config);
    DenseCalculateGradient(filter->dense, dense_gradient, d_out);

    int d_W_size = filter->config.dense.input_size * filter->config.dense.output_size;
    int d_b_size = filter->config.dense.output_size;
    int ts = filter->config.ts;
    int batch = filter->training_data->config.mini_batch_size;
    P_LOOP_START(batch, b)
        for (int t = 0; t < ts; ++t){
            op_vec_add(gradient->d_W + b * d_W_size, dense_gradient->d_W + (t * d_W_size + b * ts * d_W_size), gradient->d_W + b * d_W_size, d_W_size);
            op_vec_add(gradient->d_b + b * d_b_size, dense_gradient->d_b + (t * d_b_size + b * ts * d_b_size), gradient->d_b + b * d_b_size, d_b_size);
        }
    P_LOOP_END
    memcpy(gradient->d_X, dense_gradient->d_X, filter->config.dense.input_size * ts * batch * sizeof(float));
}

void TimeDistributedDenseDestroy(TimeDistributedDense filter){
    DenseDestroy(filter->dense);
    free(filter);
}
