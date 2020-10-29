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
#include "string.h"

typedef struct {
    BatchNormTrainingConfig config;
    float *mean;
    float *variance;
    float *input_transposed;
    float *transposed_minus_mean;
    float *minus_mean_squared;
    float *x_norm;
} BatchNormTrainingData;

BatchNormTrainingData* batch_norm_training_data_create(BatchNormConfig config, BatchNormTrainingConfig training_config){
    BatchNormTrainingData *data = malloc(sizeof(BatchNormTrainingData));

    int feat = config.feature_channels;
    int input_size = feat * config.count * training_config.mini_batch_size;
    int buffer_size = (2 * feat + 4 * input_size) * sizeof(float);
    data->config = training_config;
    data->mean = malloc(buffer_size);
    data->variance = data->mean + feat;
    data->input_transposed = data->variance + feat;
    data->transposed_minus_mean = data->input_transposed + input_size;
    data->minus_mean_squared = data->transposed_minus_mean + input_size;
    data->x_norm = data->minus_mean_squared + input_size;
    memset(data->mean, 0, buffer_size);
    return data;
}

void batch_norm_training_data_destroy(BatchNormTrainingData *data){
    free(data->mean);
    free(data);
}

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
    int weights_size = 4 * chan * sizeof(float);
    float *weights = malloc(weights_size);
    filter->weights->gamma = weights;
    filter->weights->beta = filter->weights->gamma + chan;
    filter->weights->moving_mean = filter->weights->beta + chan;
    filter->weights->moving_variance = filter->weights->moving_mean + chan;
    memset(weights, 0, weights_size);
    return filter;
}

void BatchNormDestroy(BatchNorm filter) {
    free(filter->weights->gamma);
    free(filter->weights);
    if (filter->training_data != NULL){
        batch_norm_training_data_destroy(filter->training_data);
    }
    free(filter);
}

BatchNormGradient *BatchNormGradientCreate(BatchNormConfig config, BatchNormTrainingConfig training_config) {
    BatchNormGradient *grad = malloc(sizeof(BatchNormGradient));
    int feat = config.feature_channels;
    int buff = (2 * feat + feat * config.count * training_config.mini_batch_size) * sizeof(float);
    grad->d_beta = malloc(buff);
    grad->d_gamma = grad->d_beta + feat;
    grad->d_x = grad->d_gamma + feat;
    memset(grad->d_beta, 0, buff);
    return grad;
}

void BatchNormGradientDestroy(BatchNormGradient *grad) {
    free(grad->d_beta);
    free(grad);
}

BatchNorm BatchNormCreateForTraining(BatchNormConfig config, BatchNormTrainingConfig training_config) {
    BatchNorm filter = BatchNormCreateForInference(config);
    filter->training_data = batch_norm_training_data_create(config, training_config);
    return filter;
}

BatchNormTrainingConfig BatchNormTrainingConfigCreate(float momentum, int mini_batch_size) {
    BatchNormTrainingConfig result;
    result.mini_batch_size = mini_batch_size;
    result.momentum = momentum;
    return result;
}


//output_image = (input_image - moving_mean[c]) * gamma[c] / sqrt(moving_variance[c] + epsilon) + beta[c];


void normalize(
    const float *input,
    const float *mean,
    const float *variance,
    float *buffer,
    float epsilon,
    int size,
    float *output
){
    // output = input - moving_mean
    op_vec_sub(input, mean, output, size);
    // buffer = moving_variance + epsilon
    op_vec_add_sc(variance, epsilon, buffer, size);
//    buffer = sqrt(buffer)
    op_vec_sqrt(buffer, buffer, size);
    // output = output / buffer
    op_vec_div(output, buffer, output, size);
}

void norm_mul_trainable(
    const float *input,
    const float *gamma,
    const float *beta,
    int size,
    float *output
){
    // output = input * gamma
    op_vec_mul(input, gamma, output, size);
//    output = output + beta
    op_vec_add(output, beta, output, size);
}

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
    normalize(input, mean, variance, buffer, epsilon, size, output);
    norm_mul_trainable(output, gamma, beta, size, output);
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

    P_LOOP_START(n, i)
        float buffer[feat];
        normalize(input + i * feat, mean, variance, buffer, filter->config.epsilon, feat, filter->training_data->x_norm + i * feat);
        norm_mul_trainable(filter->training_data->x_norm + i * feat, filter->weights->gamma, filter->weights->beta, feat, output + i * feat);
    P_LOOP_END

    //moving mean calculation
    float buffer[feat];

    op_vec_mul_sc(mean, 1 - momentum, buffer, feat);
    op_vec_mul_sc(filter->weights->moving_mean, momentum, filter->weights->moving_mean, feat);
    op_vec_add(buffer, filter->weights->moving_mean, filter->weights->moving_mean, feat);

    // moving variance calc
    op_vec_mul_sc(variance, 1 - momentum, buffer, feat);
    op_vec_mul_sc(filter->weights->moving_variance, momentum, filter->weights->moving_variance, feat);
    op_vec_add(buffer, filter->weights->moving_variance, filter->weights->moving_variance, feat);

    return 0;
}

void BatchNormCalculateGradient(BatchNorm filter, BatchNormGradient *gradient, float *d_out) {
    int feat = filter->config.feature_channels;
    int n = filter->config.count * filter->training_data->config.mini_batch_size;
    /* Forward
     * out = gamma(D,) * x_norm(N, D) + beta(D,);
     *
     * d_out = (N, F), F = feature channels;
     * d_beta = sum(0; F)(transpose(d_out));
     *
     * d_gamma = sum(0; F)(d_out * x_norm)
     * */

    float* buffer;

    float *transposed_d_out = buffer;

    op_mat_transp(d_out, transposed_d_out, feat, n);
    P_LOOP_START(feat, f)
        op_vec_sum(transposed_d_out + f * n, gradient->d_beta + f, n);
    P_LOOP_END

    // d_out_x_norm = d_out * x_norm
    float* d_out_x_norm = transposed_d_out + feat * n;
    op_vec_mul(d_out, filter->training_data->x_norm, d_out_x_norm, feat * n);
    // transpose(d_out_x_norm)
    float* d_out_x_norm_transposed = d_out_x_norm + feat * n;
    op_mat_transp(d_out_x_norm, d_out_x_norm_transposed, feat, n);
    // sum(0; F)(transpose(dout_x_norm))
    P_LOOP_START(feat, f)
        op_vec_sum(d_out_x_norm_transposed + f * n, gradient->d_gamma + f, n);
    P_LOOP_END

    /*
     * d_x_norm = d_out * gamma
     *
     */
    float* d_x_norm = d_out_x_norm_transposed + feat * n;
    P_LOOP_START(n, i)
        op_vec_mul(d_out + i * feat, filter->weights->gamma, d_x_norm + i * feat, feat);
    P_LOOP_END
    /*
     * x_norm = (x - mean) / sqrt(variance + epsilon)
     * x_norm = (x - mean) * 1 / sqrt(variance + epsilon)
     * xm = x - mean
     * d_xm =  d_x_norm * 1 / sqrt(variance + epsilon)
     */


}






