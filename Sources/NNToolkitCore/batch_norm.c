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

    float *x_mu;
    float *var_eps;
    float *sqrt_var;
    float *x_norm;
    float *gamma_x_norm;
} BatchNormTrainingData;

BatchNormTrainingData* batch_norm_training_data_create(BatchNormConfig config, BatchNormTrainingConfig training_config){
    BatchNormTrainingData *data = malloc(sizeof(BatchNormTrainingData));
    int feat = config.feature_channels;
    int N = config.count * training_config.mini_batch_size;
    int buffer_size = (2 * feat + 8 * N * feat) * sizeof(float);
    data->config = training_config;
    data->mean = malloc(buffer_size);
    data->variance = data->mean + feat;
    data->input_transposed = data->variance + feat;
    data->transposed_minus_mean = data->input_transposed + N * feat;
    data->minus_mean_squared = data->transposed_minus_mean + N * feat;
    data->x_mu = data->minus_mean_squared + N * feat;
    data->var_eps = data->x_mu + N * feat;
    data->sqrt_var = data->var_eps + N * feat;
    data->x_norm = data->sqrt_var + N * feat;
    data->gamma_x_norm = data->x_norm + N * feat;
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
    int buff = (2 * feat * training_config.mini_batch_size + feat * config.count * training_config.mini_batch_size) * sizeof(float);
    grad->d_beta = malloc(buff);
    grad->d_gamma = grad->d_beta + feat * training_config.mini_batch_size;
    grad->d_x = grad->d_gamma + feat * training_config.mini_batch_size;
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



typedef struct {
    float *x_mu;
    float *var_eps;
    float *sqrt_var;
    float *x_norm;
    float *gamma_x_norm;
} batch_norm_buff;

//buffer [x_mu, var_eps, sqrt_var, x_norm, gamma_x_norm]

void batch_norm(
        const float *input,
        const float *mean,
        const float *variance,
        const float *gamma,
        const float *beta,
        float *output,
        batch_norm_buff buffer,
        float epsilon,
        int size
) {
    // output = input - moving_mean
    op_vec_sub(input, mean, buffer.x_mu, size);
    // buffer = moving_variance + epsilon
    op_vec_add_sc(variance, epsilon, buffer.var_eps, size);
//    buffer = sqrt(buffer)
    op_vec_sqrt(buffer.var_eps, buffer.sqrt_var, size);
    // output = output / buffer
    op_vec_div(buffer.x_mu, buffer.sqrt_var, buffer.x_norm, size);
    // output = input * gamma
    op_vec_mul(buffer.x_norm, gamma, buffer.gamma_x_norm, size);
//    output = output + beta
    op_vec_add(buffer.gamma_x_norm, beta, output, size);
}


int BatchNormApplyInference(BatchNorm filter, const float *input, float* output) {
    if(filter->training_data != NULL){
        return -1;
    }
    P_LOOP_START(filter->config.count, index)
        size_t offset = index * filter->config.feature_channels;
        int size = filter->config.feature_channels;
        float buffer[5 * size];
        batch_norm_buff batch_norm_buff =
                { buffer, buffer + size, buffer + 2 * size, buffer + 3 * size, buffer + 4 * size };
        batch_norm(
            input + offset,
            filter->weights->moving_mean,
            filter->weights->moving_variance,
            filter->weights->gamma,
            filter->weights->beta,
            output + offset,
            batch_norm_buff,
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
    const int N = filter->config.count * filter->training_data->config.mini_batch_size;

    float *transposed = filter->training_data->input_transposed;
    op_mat_transp(input, transposed, feat, N);

    // MEAN
    float *mean = filter->training_data->mean;
    P_LOOP_START(feat, f)
        op_vec_sum(transposed + f * N, mean + f, N);
    P_LOOP_END
    op_vec_div_sc(mean, (float) N, mean, N);

    // VARIANCE
    float *variance = filter->training_data->variance;
    P_LOOP_START(feat, f)
        op_vec_add_sc(transposed + f * N, -mean[f],
                      filter->training_data->transposed_minus_mean + f * N, N);
        op_vec_mul(filter->training_data->transposed_minus_mean + f * N,
                   filter->training_data->transposed_minus_mean + f * N,
                   filter->training_data->minus_mean_squared + f * N,
                   N);
        op_vec_sum(filter->training_data->minus_mean_squared + f * N, variance + f, N);
    P_LOOP_END
    op_vec_div_sc(variance, (float)N, variance, N);

    //BATCH_NORM
    S_LOOP_START(N, n)
        batch_norm_buff buffer = {
            filter->training_data->x_mu + n * feat,
            filter->training_data->var_eps + n * feat,
            filter->training_data->sqrt_var + n * feat,
            filter->training_data->x_norm + n * feat,
            filter->training_data->gamma_x_norm + n * feat,
        };
        batch_norm(
            input + n * feat,
            mean,
            variance,
            filter->weights->gamma,
            filter->weights->beta,
            output + n * feat,
            buffer,
            filter->config.epsilon,
            feat
        );

    S_LOOP_END

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

#define PRINT 1

void BatchNormCalculateGradient(BatchNorm filter, BatchNormGradient *gradient, float *d_out) {
    int feat = filter->config.feature_channels;
    int N = filter->config.count * filter->training_data->config.mini_batch_size;
    /* Forward
     * out = gamma(D,) * x_norm(N, D) + beta(D,);
     *
     * d_out = (N, F), F = feature channels;
     * d_beta = sum(0; F)(transpose(d_out));
     *
     * d_gamma = sum(0; F)(d_out * x_norm)
     * */

    int buffer_size = (4 * feat + 12 * N * feat) * sizeof(float);

    float* buffer = malloc(buffer_size);
    memset(buffer, 0, buffer_size);

    float *transposed_d_out = buffer; //  N * feat
    //d_beta:
    op_mat_transp(d_out, transposed_d_out, feat, N);
    P_LOOP_START(feat, f)
        op_vec_sum(transposed_d_out + f * N, gradient->d_beta + f, N);
    P_LOOP_END
    // d_gamma:
    // d_out_x_norm = d_out * x_norm
    float* d_out_x_norm = transposed_d_out + feat * N;
    op_vec_mul(d_out, filter->training_data->x_norm, d_out_x_norm, feat * N);
    // transpose(d_out_x_norm)
    float* d_out_x_norm_transposed = d_out_x_norm + feat * N;
    op_mat_transp(d_out_x_norm, d_out_x_norm_transposed, feat, N);
    // sum(0; F)(transpose(dout_x_norm))
    P_LOOP_START(feat, f)
        op_vec_sum(d_out_x_norm_transposed + f * N, gradient->d_gamma + f, N);
    P_LOOP_END

    /*
     * d_x_norm = d_out * gamma
     *
     */

    float* d_x_norm = d_out_x_norm_transposed + feat * N;
    P_LOOP_START(N, n)
        op_vec_mul(d_out + n * feat, filter->weights->gamma, d_x_norm + n * feat, feat);
    P_LOOP_END
    /*
     * x_norm = (x - mean) / sqrt(variance + epsilon)
     * x_norm = (x - mean) * 1 / sqrt(variance + epsilon)
     * x_m = x - mean
     * d_xm1 =  d_x_norm / sqrt(variance + epsilon)
     */
    float *d_x_mu_1 = d_x_norm + feat * N;
    op_vec_div(d_x_norm, filter->training_data->sqrt_var, d_x_mu_1, feat * N);
    /*
     * ivar = 1 / sqrt_var
     * d_ivar =  Sum(d_x_norm * x_mu)
     * */
    float *d_x_norm_mu = d_x_mu_1 + 2 * feat * N;

    op_vec_mul(d_x_norm, filter->training_data->x_mu, d_x_norm_mu, feat * N);

    float *d_x_norm_mu_transp = d_x_norm_mu + feat * N;

    op_mat_transp(d_x_norm_mu, d_x_norm_mu_transp, feat, N);

    float *d_ivar = d_x_norm_mu + feat * N;

    P_LOOP_START(feat, f)
        op_vec_sum(d_x_norm_mu_transp + f * N,d_ivar + f, N);
    P_LOOP_END

    /*
     * d_sqrtvar = - divar / (var_eps)
     */

    float *d_sqrt_var = d_ivar + feat;

    op_vec_mul_sc(d_ivar, -1.0f, d_sqrt_var, feat);
    op_vec_div(d_sqrt_var, filter->training_data->var_eps, d_sqrt_var, feat);
    /*
     * d_var = dsqrt_var / 2 * sqrtvar
     */

    float *d_var = d_sqrt_var + feat;

    op_vec_div(d_sqrt_var, filter->training_data->sqrt_var, d_var, feat);
    op_vec_div_sc(d_var, 2.0f, d_var, feat);

    // d_var = d_var / N; for the next step
    op_vec_div_sc(d_var, (float)N, d_var, feat);
    /*
     * d_x_mu_2 = 2 * x_m * (d_var / N)
     */

    float *d_x_mu_2 = d_var + feat;

    op_vec_mul_sc(filter->training_data->x_mu, 2, d_x_mu_2, N * feat);

    P_LOOP_START(N, n)
        op_vec_mul(d_x_mu_2 + n * feat, d_var, d_x_mu_2 + n * feat, feat);
    P_LOOP_END

    float *d_x_mu = d_x_mu_2 + N * feat;
    // d_x_mu = 1 + 2
    op_vec_add(d_x_mu_1, d_x_mu_2, d_x_mu, N * feat);

    float *d_x_1 = d_x_mu + N * feat;

    memcpy(d_x_1, d_x_mu, N * feat * sizeof(float));

    /*
     * d_mu = - 1 * sum(1, N)(d_x_mu)
     */

    float *d_x_mu_transposed = d_x_1 + N * feat;

    op_mat_transp(d_x_mu, d_x_mu_transposed, feat, N);

    float *d_mu = d_x_mu_transposed + N * feat;

    P_LOOP_START(feat, f)
        op_vec_sum(d_x_mu_transposed + f * N,d_mu + f, N);
    P_LOOP_END
    op_vec_mul_sc(d_mu, (-1.0f / (float )N), d_mu, feat);
    float *d_x_2 = d_mu + feat;
    S_LOOP_START(N, n)
        memcpy(d_x_2 + n * feat, d_mu, feat * sizeof(float));
    S_LOOP_END
    op_vec_add(d_x_1, d_x_2, gradient->d_x, N * feat);
    free(buffer);
}






