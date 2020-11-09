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
#include "loops.h"


typedef struct{
    DenseTrainingConfig config;
    float *x;
    float *z;
    float *a;
    float *dz;
} DenseTrainingData;

DenseTrainingData* dense_training_data_create(DenseConfig config, DenseTrainingConfig training_config){
    DenseTrainingData *data = malloc(sizeof(DenseTrainingData));
    data->config = training_config;
    int x_size = config.input_size * training_config.mini_batch_size;
    int z_size = config.output_size * training_config.mini_batch_size;
    int buff_size = (x_size + 3 * z_size) * sizeof(float);
    data->x = malloc(buff_size);
    data->z = data->x + x_size;
    data->a = data->z + z_size;
    data->dz = data->a + z_size;
    memset(data->x, 0, buff_size);
    return data;
}

void dense_training_data_destroy(DenseTrainingData *data){
    free(data->x);
    free(data);
}

struct DenseStruct {
    DenseConfig config;
    DenseTrainingData* training_data;
    DenseWeights* weights;
};

DenseWeights* DenseGetWeights(Dense filter){
    return filter->weights;
}

DenseConfig DenseConfigCreate(int input_size, int output_size, ActivationFunction activation){
    DenseConfig config;
    config.input_size = input_size;
    config.output_size = output_size;
    config.activation = activation;
    return config;
}

Dense DenseCreateForInference(DenseConfig config) {
    Dense filter = malloc(sizeof(struct DenseStruct));
    filter->config = config;
    filter->training_data = NULL;
    filter->weights = malloc(sizeof(DenseWeights));
    int weights_buff_size = config.input_size * (config.output_size + 1) * sizeof(float);
    filter->weights->W = malloc(weights_buff_size);
    filter->weights->b = filter->weights->W + config.input_size * config.output_size;
    memset(filter->weights->W, 0, weights_buff_size);
    return filter;
}

void DenseDestroy(Dense filter) {
    free(filter->weights->W);
    free(filter->weights);
    if (filter->training_data){
        dense_training_data_destroy(filter->training_data);
    }
    free(filter);
}

DenseTrainingConfig DenseTrainingConfigCreate(int batch){
    DenseTrainingConfig config;
    config.mini_batch_size = batch;
    return config;
}

Dense DenseCreateForTraining(DenseConfig config, DenseTrainingConfig training_config) {
    Dense filter = DenseCreateForInference(config);
    filter->training_data = dense_training_data_create(config, training_config);
    return filter;
}

DenseGradient* DenseGradientCreate(DenseConfig config, DenseTrainingConfig training_config) {
    DenseGradient* grad = malloc(sizeof(DenseGradient));
    int d_w_size = config.input_size * config.output_size * training_config.mini_batch_size;
    int d_x_size = config.input_size * training_config.mini_batch_size;
    int buff_size = (d_w_size + d_x_size + config.output_size * training_config.mini_batch_size) * sizeof(float);
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

DenseConfig DenseGetConfig(Dense filter) {
    return filter->config;
}

void z(Dense filter, const float *input, float* output){
    op_mat_mul(input, filter->weights->W, output, 1, filter->config.output_size, filter->config.input_size);
    op_vec_add(output, filter->weights->b, output, filter->config.output_size);
}

void a(Dense filter, const float *input, float* output){
    if (filter->config.activation) {
        ActivationFunctionApply(filter->config.activation, input, output);
    } else if (input != output){
        memcpy(output, input, filter->config.output_size * sizeof(float));
    }
}

int DenseApplyInference(Dense filter, const float *input, float* output) {
    if(filter->training_data != NULL){
        return -1;
    }
    z(filter, input, output);
    a(filter, output, output);
    return 0;
}

int DenseApplyTrainingBatch(Dense filter, const float *input, float* output) {
    if (filter->training_data == NULL){
        return -1;
    }
    
    int in = filter->config.input_size;
    int batch = filter->training_data->config.mini_batch_size;
    int out = filter->config.output_size;

    memcpy(filter->training_data->x, input, in * batch * sizeof(float));

    P_LOOP_START(batch, b)
        z(filter, input + b * in, filter->training_data->z + b * out);
        a(filter, filter->training_data->z + b * out, filter->training_data->a + b * out);
    P_LOOP_END

    memcpy(output, filter->training_data->a, out * batch * sizeof(float));
    return 0;
}

void DenseCalculateGradient(Dense filter, DenseGradient *gradient, float *d_out) {
    int out = filter->config.output_size;
    int in = filter->config.input_size;
    P_LOOP_START(filter->training_data->config.mini_batch_size, b)
        // dz = d_out * d_activation ?? 1;
        float *dz = filter->training_data->dz + b * out;
        if(filter->config.activation){
            ActivationFunctionApplyDerivative(filter->config.activation, filter->training_data->z + b * out, filter->training_data->a + b * out, dz);
            op_vec_mul(dz, d_out + b * out, dz, out);
        } else {
            memcpy(dz, d_out + b * out, out * sizeof(float));
        }
        //db = dz;
        memcpy(gradient->d_b + b * out, dz, out * sizeof(float));
        // DW = dz * X;
        op_mat_mul(filter->training_data->x + b * in, dz, gradient->d_W + b * in * out, in, out, 1);
        // DX = dz * W;
        op_mat_mul(filter->weights->W, dz, gradient->d_X + b * in, in, 1, out);
    P_LOOP_END
}

