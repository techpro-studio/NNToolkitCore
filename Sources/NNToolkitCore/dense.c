//
//  dense.c
//  audio_test
//
//  Created by Alex on 29.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "dense.h"
#include "operations.h"
#include "stdlib.h"
#include "string.h"

typedef struct{
    DenseTrainingConfig config;
    float *x;
    float *z;
    float *a;
    float *dz;
} DenseTrainingData;

DenseTrainingData* DenseTrainingDataCreate(DenseConfig config, DenseTrainingConfig training_config){
    DenseTrainingData *data = malloc(sizeof(DenseTrainingData));
    data->config = training_config;
    int x_size = config.inputSize * training_config.mini_batch_size;
    int z_size = config.outputSize * training_config.mini_batch_size;
    int buff_size = (x_size + 3 * z_size) * sizeof(float);
    data->x = malloc(buff_size);
    data->z = data->x + x_size;
    data->a = data->z + z_size;
    data->dz = data->a + z_size;
    memset(data->x, 0, buff_size);
    return data;
}

void DenseTrainingDataDestroy(DenseTrainingData *data){
    free(data->x);
    free(data);
}

struct DenseFilterStruct {
    DenseConfig config;
    DenseTrainingData* traning_data;
    DenseWeights* weights;
};

DenseWeights* DenseFilterGetWeights(DenseFilter filter){
    return filter->weights;
}

DenseConfig DenseConfigCreate(int inputSize, int outputSize, ActivationFunction activation){
    DenseConfig config;
    config.inputSize = inputSize;
    config.outputSize = outputSize;
    config.activation = activation;
    return config;
}

DenseFilter DenseFilterCreateForInference(DenseConfig config) {
    DenseFilter filter = malloc(sizeof(struct DenseFilterStruct));
    filter->config = config;
    filter->traning_data = NULL;
    filter->weights = malloc(sizeof(DenseWeights));
    int weights_buff_size = config.inputSize * (config.outputSize + 1) * sizeof(float);
    filter->weights->W = malloc(weights_buff_size);
    filter->weights->b = filter->weights->W + config.inputSize * config.outputSize;
    memset(filter->weights->W, 0, weights_buff_size);
    return filter;
}

void DenseFilterDestroy(DenseFilter filter) {
    free(filter->weights->W);
    free(filter->weights);
    if (filter->traning_data){
        DenseTrainingDataDestroy(filter->traning_data);
    }
    free(filter);
}

DenseTrainingConfig DenseTrainingConfigCreate(int batch){
    DenseTrainingConfig config;
    config.mini_batch_size = batch;
    return config;
}

DenseFilter DenseFilterCreateForTraining(DenseConfig config, DenseTrainingConfig training_config) {
    DenseFilter filter = DenseFilterCreateForInference(config);
    filter->traning_data = DenseTrainingDataCreate(config, training_config);
    return filter;
}

DenseGradient* DenseGradientCreate(DenseConfig config, DenseTrainingConfig trainingConfig) {
    DenseGradient* grad = malloc(sizeof(DenseGradient));
    int d_w_size = config.inputSize * config.outputSize * trainingConfig.mini_batch_size;
    int d_x_size = config.inputSize * trainingConfig.mini_batch_size;
    int buff_size = (d_w_size + d_x_size + config.outputSize * trainingConfig.mini_batch_size) * sizeof(float);
    grad->d_W = (float *) malloc(buff_size);
    grad->d_X = grad->d_W + d_w_size;
    grad->d_b = grad->d_X + d_x_size;
    memset(grad->d_W, 0, buff_size);
    return grad;
}

void DenseGradientDestroy(DenseGradient *gradient){
    free(gradient->d_W);
    free(gradient);
}

DenseConfig DenseFilterGetConfig(DenseFilter filter) {
    return filter->config;
}

void z(DenseFilter filter, const float *input, float* output){
    op_mat_mul(input, filter->weights->W, output, 1, filter->config.outputSize, filter->config.inputSize, 0.0);
    op_vec_add(output, filter->weights->b, output, filter->config.outputSize);
}

void a(DenseFilter filter, const float *input, float* output){
    if (filter->config.activation) {
        ActivationFunctionApply(filter->config.activation, input, output);
    } else if (input != output){
        memcpy(output, input, filter->config.outputSize * sizeof(float));
    }
}

int DenseFilterApply(DenseFilter filter, const float *input, float* output) {
    z(filter, input, output);
    a(filter, output, output);
    return 0;
}

int DenseFilterApplyTrainingBatch(DenseFilter filter, const float *input, float* output) {
    if (filter->traning_data == NULL){
        return -1;
    }
    
    int in = filter->config.inputSize;
    int batch = filter->traning_data->config.mini_batch_size;
    int out = filter->config.outputSize;

    memcpy(filter->traning_data->x, input, in * batch * sizeof(float));

    P_LOOP_START(batch, b)
        z(filter, input + b * in, filter->traning_data->z + b * out);
        a(filter, filter->traning_data->z + b * out, filter->traning_data->a + b * out);
    P_LOOP_END

    memcpy(output, filter->traning_data->a, out * batch * sizeof(float));
    return 0;
}

void DenseFilterCalculateGradient(DenseFilter filter, DenseGradient *gradient, float *d_out) {
    int out = filter->config.outputSize;
    int in = filter->config.inputSize;
    P_LOOP_START(filter->traning_data->config.mini_batch_size, b)
        // dz = d_out * d_activation ?? 1;
        float *dz = filter->traning_data->dz + b * out;
        if(filter->config.activation){
            ActivationFunctionApplyDerivative(filter->config.activation, filter->traning_data->z + b * out, filter->traning_data->a + b * out, dz);
            op_vec_mul(dz, d_out + b * out, dz, out);
        } else {
            memcpy(dz, d_out + b * out, out * sizeof(float));
        }
        //db = dz;
        memcpy(gradient->d_b + b * out, dz, out * sizeof(float));
        // DW = dz * X;
                op_mat_mul(filter->traning_data->x + b * in, dz, gradient->d_W + b * in * out, in, out, 1, 0.0);
        // DX = dz * W;
                op_mat_mul(filter->weights->W, dz, gradient->d_X + b * in, in, 1, out, 0.0);
    P_LOOP_END
}

