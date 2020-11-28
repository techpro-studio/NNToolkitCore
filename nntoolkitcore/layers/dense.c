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
#include "nntoolkitcore/layers/private/weights_private.h"

typedef struct {
    DenseTrainingConfig config;
    float *x;
    float *z;
    float *a;
    float *dz;
    DefaultGradient **batch_gradients;
} DenseTrainingData;

DenseWeightsSize dense_weight_size_from_config(DenseConfig config){
    int w_size = config.input_size * config.output_size;
    int sum = w_size + config.output_size;
    return (DefaultWeightsSize) { .w = w_size, .b = config.output_size, .sum = sum };
}

DenseTrainingData *dense_training_data_create(DenseConfig config, DenseTrainingConfig training_config) {
    DenseTrainingData *data = malloc(sizeof(DenseTrainingData));
    data->config = training_config;
    int b = training_config.mini_batch_size;
    int x_size = config.input_size * b;
    int z_size = config.output_size * b;
    int buff_size = x_size + 3 * z_size;
    data->x = f_malloc(buff_size);
    data->z = data->x + x_size;
    data->a = data->z + z_size;
    data->dz = data->a + z_size;
    data->batch_gradients = malloc( b * sizeof(DefaultGradient*));
    for (int i = 0; i < b; ++i){
        data->batch_gradients[i] = default_gradient_create(dense_weight_size_from_config(config), 0);
    }
    return data;
}

void dense_training_data_destroy(DenseTrainingData *data) {
    for (int i = 0; i < data->config.mini_batch_size; ++i){
        default_gradient_destroy(data->batch_gradients[i]);
    }
    free(data->batch_gradients);
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
    filter->weights = default_weights_create(dense_weight_size_from_config(config));
    return filter;
}

void DenseDestroy(Dense filter) {
    default_weights_destroy(filter->weights);
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
    return default_gradient_create(
        dense_weight_size_from_config(config),
        training_config.mini_batch_size * config.input_size
    );
}

DenseGradient *DenseGradientCreateFromFilter(Dense dense) {
    if (dense->training_data == NULL){
        return NULL;
    }
    return DenseGradientCreate(dense->config, dense->training_data->config);
}


void DenseGradientDestroy(DenseGradient *gradient) {
    default_gradient_destroy(gradient);
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
    int batch = filter->training_data->config.mini_batch_size;
    P_LOOP_START(batch, b)
                // dz = d_out * d_activation ?? 1;
        float *dz = filter->training_data->dz + b * out;
        if (filter->config.activation) {
            ActivationFunctionCalculateGradient(filter->config.activation, filter->training_data->z + b * out,
                                                filter->training_data->a + b * out, d_out + b * out, dz);
        } else {
            f_copy(dz, d_out + b * out, out);
        }
        //db = dz;
        f_copy(filter->training_data->batch_gradients[b]->d_b, dz, out);
        // DW = dz * X;
        op_mat_mul(filter->training_data->x + b * in, dz, filter->training_data->batch_gradients[b]->d_W, in, out, 1);
        // DX = dz * W;
        op_mat_mul(filter->weights->W, dz, gradient->d_X + b * in, in, 1, out);
    P_LOOP_END
    default_gradient_sum(filter->training_data->batch_gradients, gradient, dense_weight_size_from_config(filter->config), batch);
}


