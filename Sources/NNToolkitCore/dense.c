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
    filter->weights = malloc(sizeof(DenseWeights));
    filter->weights->W = malloc(config.inputSize * (config.outputSize + 1) * sizeof(float));
    filter->weights->b = filter->weights->W + config.inputSize * config.outputSize;
    return filter;
}

void DenseFilterDestroy(DenseFilter filter) {
    free(filter->weights->W);
    free(filter->weights);
    free(filter);
}

DenseTrainingConfig DenseTrainingConfigCreate(int batch){
    DenseTrainingConfig config;
    config.mini_batch_size = batch;
    return config;
}

DenseFilter DenseFilterCreateForTraining(DenseConfig config, DenseTrainingConfig training_config) {
    DenseFilter filter = DenseFilterCreateForInference(config);
    filter->traning_data = malloc()
    return filter;
}

DenseGradient* DenseGradientCreate(DenseConfig config, DenseTrainingConfig trainingConfig) {
    DenseGradient* grad = malloc(sizeof(DenseGradient));
    int dWSize = config.inputSize * config.outputSize * trainingConfig.mini_batch_size;
    int dXSize = config.inputSize * trainingConfig.mini_batch_size;
    grad->d_W = (float *) malloc((dWSize + dXSize + config.outputSize * trainingConfig.mini_batch_size) * sizeof(float));
    grad->d_X = grad->d_W + dWSize;
    grad->d_b = grad->d_X + dXSize;
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
    MatMul(input, filter->weights->W, output, 1,  filter->config.outputSize, filter->config.inputSize, 0.0);
    VectorAdd(output, filter->weights->b, output, filter->config.outputSize);
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
    a(filter, input, output);
    return 0;
}

int DenseFilterApplyTrainingBatch(DenseFilter filter, const float *input, float* output) {
    if (filter->traning_data == NULL){
        return -1;
    }
    memcpy(filter->traning_data->x, input, filter->config.inputSize * filter->traning_data->config.mini_batch_size);
    P_LOOP_START(filter->traning_data->config.mini_batch_size, b)
        z(filter, input + b * filter->config.inputSize, filter->traning_data->z + b * filter->config.outputSize);
        a(filter, filter->traning_data->z + b * filter->config.outputSize, filter->traning_data->a + b * filter->config.outputSize);
        memcpy(output+ b * filter->config.outputSize, filter->traning_data->a + b * filter->config.outputSize, filter->config.outputSize * sizeof(float));
    P_LOOP_END
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
            VectorMul(dz, d_out + b * out, dz, out);
        } else {
            memcpy(dz, d_out + b * out, out);
        }
        //db = dz;
        memcpy(dz, gradient->d_b + b * out, out);
        // DW = dz * X;
        MatMul(filter->traning_data->x + b * in, dz, gradient->d_W, in, out, 1, 0.0);
        // DX = dz * W;
        MatMul(filter->weights->W, dz, gradient->d_X, in, 1, out, 0.0);
    P_LOOP_END
}

