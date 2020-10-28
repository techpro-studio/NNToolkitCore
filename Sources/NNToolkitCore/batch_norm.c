//
//  batch_norm.c
//  audio_test
//
//  Created by Alex on 24.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "batch_norm.h"
#import "operations.h"
#include "stdlib.h"

typedef struct {
    BatchNormTrainingConfig config;
    float *mean;
    float *input_transposed;
    float *transposed_minus_mean;
    float *minus_mean_squared;
    float *variance;
} BatchNormTrainingData;

struct BatchNormFilterStruct{
    BatchNormTrainingData *training_data;
    BatchNormConfig config;
    BatchNormWeights* weights;
};

BatchNormWeights* BatchNormGetWeights(BatchNorm filter){
    return filter->weights;
}

BatchNormConfig BatchNormConfigCreate(int feature_channels, float epsilon, int count){
    BatchNormConfig config;
    config.count = count;
    config.epsilon = epsilon;
    config.feature_channels = feature_channels;
    return config;
}

BatchNorm BatchNormCreateForInference(BatchNormConfig config) {
    BatchNorm filter = malloc(sizeof(struct BatchNormFilterStruct));
    filter->config = config;
    filter->training_data = NULL;
    int chan = config.feature_channels;
    filter->weights = malloc(sizeof(BatchNormWeights));
    float *weights = malloc(4 * chan * sizeof(float));
    filter->weights->gamma = weights;
    filter->weights->beta = filter->weights->gamma + chan;
    filter->weights->moving_mean = filter->weights->beta + chan;
    filter->weights->moving_variance = filter->weights->moving_mean + chan;
    return filter;
}

void BatchNormDestroy(BatchNorm filter) {
    free(filter->weights->gamma);
    free(filter->weights);
    free(filter);
}


//output_image = (input_image - moving_mean[c]) * gamma[c] / sqrt(moving_variance[c] + epsilon) + beta[c];


void batch_norm(
        const float *input,
        const float *mean,
        const float *variance,
        const float *gamma,
        const float *beta,
        float *output,
        float *buffer,
        float epsilon,
        int size
) {
    // output = input - moving_mean
    op_vec_sub(input, mean, output, size);
    // output = (input - moving_mean) * gamma
    op_vec_mul(output, gamma, output, size);
    // buffer = moving_variance + epsilon
    op_vec_add_sc(variance, epsilon, buffer, size);
//    buffer = sqrt(buffer)
    op_vec_sqrt(buffer, buffer, size);
    // output = output / buffer
    op_vec_div(output, buffer, output, size);
//    output = output + beta
    op_vec_add(output, beta, output, size);
}


int BatchNormApplyInference(BatchNorm filter, const float *input, float* output) {
    if(filter->training_data != NULL){
        return -1;
    }
    P_LOOP_START(filter->config.count, index)
        size_t offset = index * filter->config.feature_channels;
        int size = filter->config.feature_channels;
        float buffer[size];
        batch_norm(
                input + offset,
                filter->weights->moving_mean,
                filter->weights->moving_variance,
                filter->weights->gamma,
                filter->weights->beta,
                output + offset,
                buffer,
                filter->config.epsilon,
                size
        );
    P_LOOP_END
    return 0;
}

int BatchNormApplyTrainingBatch(BatchNorm filter, const float *input, float *output) {
    if (filter->training_data == NULL){
        return -1;
    }

    const int batch = filter->training_data->config.mini_batch_size;
    const float momentum = filter->training_data->config.momentum;
    const int feat = filter->config.feature_channels;
    const int n = filter->config.count * filter->training_data->config.mini_batch_size;

    float *transposed = filter->training_data->input_transposed;
    op_mat_transp(input, transposed, feat, n);

    // SUM()
    float *mean = filter->training_data->mean;
    P_LOOP_START(feat, f)
        op_vec_sum(transposed + f * n, mean + f, n);
    P_LOOP_END
    op_vec_div_sc(mean, (float) n, mean, n);

    float *variance = filter->training_data->variance;

    P_LOOP_START(feat, f)
        op_vec_add_sc(transposed + f * n, -mean[f],
                      filter->training_data->transposed_minus_mean + f * n, n);
        op_vec_mul(filter->training_data->transposed_minus_mean + f * n,
                   filter->training_data->transposed_minus_mean + f * n,
                   filter->training_data->minus_mean_squared + f * n,
                   n);
        op_vec_sum(filter->training_data->minus_mean_squared + f * n, variance + f, n);
    P_LOOP_END
    op_vec_div_sc(variance, (float) n, variance, n);

    
    return 0;
}

BatchNormGradient *BatchNormGradientCreate(BatchNormConfig config, BatchNormTrainingConfig training_config) {
    return NULL;
}

BatchNorm BatchNormCreateForTraining(BatchNormConfig config, BatchNormTrainingConfig training_config) {
    return NULL;
}

BatchNormTrainingConfig BatchNormTrainingConfigCreate(float momentum, int mini_batch_size) {
    BatchNormTrainingConfig result;
    result.mini_batch_size = mini_batch_size;
    result.momentum = momentum;
    return result;
}


