//
//  dense.c
//  audio_test
//
//  Created by Alex on 29.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "nntoolkitcore/layers/dense.h"
#include "nntoolkitcore/core/ops.h"
#include "nntoolkitcore/core/memory.h"
#include "nntoolkitcore/core/loop.h"

typedef struct {
    DenseTrainingConfig config;
    float *x;
    float *z;
    float *a;
    float *dz;
} DenseTrainingData;

DenseTrainingData *dense_training_data_create(DenseConfig config, DenseTrainingConfig training_config) {
    DenseTrainingData *data = malloc(sizeof(DenseTrainingData));
    data->config = training_config;
    int x_size = config.input_size * training_config.mini_batch_size;
    int z_size = config.output_size * training_config.mini_batch_size;
    int buff_size = x_size + 3 * z_size;
    data->x = f_malloc(buff_size);
    data->z = data->x + x_size;
    data->a = data->z + z_size;
    data->dz = data->a + z_size;
    return data;
}

void dense_training_data_destroy(DenseTrainingData *data) {
    free(data->x);
    free(data);
}

struct DenseStruct {
    DenseConfig config;
    DenseTrainingData *training_data;
    DenseWeights *weights;
};

DenseWeights *DenseGetWeights(Dense filter) {
    return filter->weights;
}

DenseConfig DenseConfigCreate(int input_size, int output_size, ActivationFunction activation) {
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
    int weights_size = config.input_size * (config.output_size + 1);
    filter->weights->W = f_malloc(weights_size);
    filter->weights->b = filter->weights->W + config.input_size * config.output_size;
    return filter;
}

void DenseDestroy(Dense filter) {
    free(filter->weights->W);
    free(filter->weights);
    if (filter->training_data) {
        dense_training_data_destroy(filter->training_data);
    }
    free(filter);
}

Dense DenseCreateForTraining(DenseConfig config, DenseTrainingConfig training_config) {
    Dense filter = DenseCreateForInference(config);
    filter->training_data = dense_training_data_create(config, training_config);
    return filter;
}

DenseGradient *DenseGradientCreate(DenseConfig config, DenseTrainingConfig training_config) {
    DenseGradient *grad = malloc(sizeof(DenseGradient));
    int d_w_size = config.input_size * config.output_size * training_config.mini_batch_size;
    int d_x_size = config.input_size * training_config.mini_batch_size;
    int grad_size = d_w_size + d_x_size + config.output_size * training_config.mini_batch_size;
    grad->d_W = f_malloc(grad_size);
    grad->d_X = grad->d_W + d_w_size;
    grad->d_b = grad->d_X + d_x_size;
    return grad;
}

void DenseGradientDestroy(DenseGradient *gradient) {
    free(gradient->d_W);
    free(gradient);
}

DenseConfig DenseGetConfig(Dense filter) {
    return filter->config;
}

void z(Dense filter, const float *input, float *output) {
    op_mat_mul(input, filter->weights->W, output, 1, filter->config.output_size, filter->config.input_size);
    op_vec_add(output, filter->weights->b, output, filter->config.output_size);
}

void a(Dense filter, const float *input, float *output) {
    if (filter->config.activation) {
        ActivationFunctionApply(filter->config.activation, input, output);
    } else if (input != output) {
        f_copy(output, input, filter->config.output_size);
    }
}

int DenseApplyInference(Dense filter, const float *input, float *output) {
    if (filter->training_data != NULL) {
        return -1;
    }
    z(filter, input, output);
    a(filter, output, output);
    return 0;
}

int DenseApplyTrainingBatch(Dense filter, const float *input, float *output) {
    if (filter->training_data == NULL) {
        return -1;
    }

    int in = filter->config.input_size;
    int batch = filter->training_data->config.mini_batch_size;
    int out = filter->config.output_size;

    f_copy(filter->training_data->x, input, in * batch);

    P_LOOP_START(batch, b)
        z(filter, input + b * in, filter->training_data->z + b * out);
        a(filter, filter->training_data->z + b * out, filter->training_data->a + b * out);
    P_LOOP_END

    f_copy(output, filter->training_data->a, out * batch);
    return 0;
}

void DenseCalculateGradient(Dense filter, DenseGradient *gradient, float *d_out) {
    int out = filter->config.output_size;
    int in = filter->config.input_size;
    P_LOOP_START(filter->training_data->config.mini_batch_size, b)
                // dz = d_out * d_activation ?? 1;
        float *dz = filter->training_data->dz + b * out;
        if (filter->config.activation) {
            ActivationFunctionCalculateGradient(filter->config.activation, filter->training_data->z + b * out,
                                                filter->training_data->a + b * out, d_out + b * out, dz);
        } else {
            f_copy(dz, d_out + b * out, out);
        }
        //db = dz;
        f_copy(gradient->d_b + b * out, dz, out);
        // DW = dz * X;
        op_mat_mul(filter->training_data->x + b * in, dz, gradient->d_W + b * in * out, in, out, 1);
        // DX = dz * W;
        op_mat_mul(filter->weights->W, dz, gradient->d_X + b * in, in, 1, out);
    P_LOOP_END
}

