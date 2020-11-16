//
//  conv_1d.c
//  mac_test
//
//  Created by Alex on 25.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "nntoolkitcore/layers/conv_1d.h"
#include "nntoolkitcore/core/loop.h"
#include "nntoolkitcore/core/ops.h"
#include "stdlib.h"
#include "string.h"


typedef struct {
    ConvTrainingConfig config;
    float *input_transposed;
} Conv1dTrainingData;

struct Conv1dFilterStruct {
    Conv1dConfig config;
    ConvWeights *weights;
    void *buffer;
    Conv1dTrainingData *training_data;
};

static Conv1dTrainingData *conv1d_training_data_create(Conv1dConfig config, ConvTrainingConfig training_config) {
    Conv1dTrainingData *data = malloc(sizeof(Conv1dTrainingData));
    data->input_transposed = malloc(config.input_feature_channels * config.input_size
             * training_config.mini_batch_size * sizeof(float));
    return data;
}

static void conv_training_data_destroy(Conv1dTrainingData *training_data){
    free(training_data->input_transposed);
    free(training_data);
}

ConvWeights *Conv1dGetWeights(Conv1d filter) {
    return filter->weights;
}

Conv1dConfig Conv1dConfigCreate(int input_feature_channels, int output_feature_channels, int kernel_size, int stride,
                                int inputSize) {
    Conv1dConfig config;
    config.input_feature_channels = input_feature_channels;
    config.output_feature_channels = output_feature_channels;
    config.input_size = inputSize;
    config.kernel_size = kernel_size;
    config.stride = stride;
    config.output_size = ((inputSize - (kernel_size - stride)) / stride);
    return config;
}

ConvTrainingConfig ConvTrainingConfigCreate(int mini_batch_size) {
    ConvTrainingConfig result;
    result.mini_batch_size = mini_batch_size;
    return result;
}

Conv1d conv1d_create(Conv1dConfig config){
    Conv1d filter = malloc(sizeof(struct Conv1dFilterStruct));
    filter->config = config;
    filter->weights = malloc(sizeof(ConvWeights));
    int W_size = config.kernel_size * config.input_feature_channels * config.output_feature_channels;
    int weights_size = (W_size + config.output_feature_channels) * sizeof(float);
    filter->weights->W = malloc(weights_size);
    filter->weights->b = filter->weights->W + W_size;
    memset(filter->weights->W, 0, weights_size);
    filter->training_data = NULL;
    filter->buffer = NULL;
    return filter;
}

Conv1d Conv1dCreateForInference(Conv1dConfig config) {
    Conv1d filter = conv1d_create(config);
    filter->buffer = malloc(config.input_size * config.input_feature_channels * sizeof(float));
    return filter;
}

Conv1d Conv1dCreateForTraining(Conv1dConfig config, ConvTrainingConfig training_config) {
    Conv1d filter = conv1d_create(config);
    filter->training_data = conv1d_training_data_create(config, training_config);
    return filter;
}

void Conv1dDestroy(Conv1d filter) {
    if (filter->buffer != NULL) {
        free(filter->buffer);
    }
    if (filter->training_data != NULL){
        conv_training_data_destroy(filter->training_data);
    }
    free(filter->weights->W);
    free(filter->weights);
    free(filter);
}

static void conv_1d_one(const struct Conv1dFilterStruct *filter, const float *input, float *output,
                 float *input_transposed_buffer) {
    float *float_input = input_transposed_buffer;
    op_mat_transp((float *) input, float_input, filter->config.input_feature_channels, filter->config.input_size);
    int k_size = filter->config.kernel_size;
    P_LOOP_START(filter->config.output_feature_channels, out_feature)
        for (int x = 0; x < filter->config.output_size; ++x) {
            int weights_offset = (int) out_feature * filter->config.input_feature_channels * k_size;
            const float *output_feature_weights = filter->weights->W + weights_offset;

            float result = 0.0f;

            int input_row_offset = x * filter->config.stride;

            for (int i = 0; i < filter->config.input_feature_channels; ++i) {
                const float *row_ptr = float_input + i * filter->config.input_size + input_row_offset;
                const float *weights_ptr = output_feature_weights + (i * k_size);
                result += op_vec_dot(row_ptr, weights_ptr, k_size);
            }

            result += filter->weights->b[out_feature];

            ((float *) output)[x * filter->config.output_feature_channels + out_feature] = result;
        }
    P_LOOP_END
}


int Conv1dApplyInference(Conv1d filter, const float *input, float *output) {
    if (filter->training_data != NULL) {
        return -1;
    }
    conv_1d_one(filter, input, output, filter->buffer);
    return 0;
}

ConvGradient *Conv1dCreateGradient(Conv1dConfig config, ConvTrainingConfig training_config) {
    ConvGradient* gradient = malloc(sizeof(ConvGradient));
    int d_x_size = config.input_size * config.input_feature_channels * training_config.mini_batch_size;
    int d_w_size = config.input_feature_channels * config.output_feature_channels * config.kernel_size;
    int grad_buffer_size = (d_x_size + d_w_size + config.output_feature_channels) * sizeof(float);
    gradient->d_W = malloc(grad_buffer_size);
    gradient->d_X = gradient->d_W + d_w_size;
    gradient->d_b = gradient->d_X + d_x_size;
    return gradient;
}

void ConvGradientDestroy(ConvGradient *gradient) {
    free(gradient->d_W);
    free(gradient);
}

int Conv1dApplyTrainingBatch(Conv1d filter, const float *input, float *output) {
    if (filter->training_data == NULL) {
        return -1;
    }
    int input_one_size = filter->config.input_size * filter->config.input_feature_channels;
    int output_one_size = filter->config.output_feature_channels * filter->config.output_size;
    int batch = filter->training_data->config.mini_batch_size;
    for (int b = 0; b < batch; ++b) {
        conv_1d_one(
            filter,
            input + b * input_one_size,
            output + b * output_one_size,
            filter->training_data->input_transposed + b * input_one_size
        );
    }
    return 0;
}

void Conv1dCalculateGradient(Conv1d filter, ConvGradient *gradient, const float *d_out) {
    int k_size = filter->config.kernel_size;
    memcpy(gradient->d_b, d_out,
       filter->config.output_size *
       filter->config.output_feature_channels *
       sizeof(float)
    );
    int batch = filter->training_data->config.mini_batch_size;
    int in = filter->config.input_feature_channels;
    int out = filter->config.output_feature_channels;
    int inp_size = in * filter->config.input_size;
    int out_size = out * filter->config.output_size;
    int dx_out_one_size = out * inp_size;
    int buffer_size = batch * dx_out_one_size + inp_size * batch * sizeof(float);

    float *d_x_out_buffer = malloc(buffer_size);
    float *d_x_transposed = d_x_out_buffer + batch * dx_out_one_size;
    memset(d_x_out_buffer, 0, buffer_size);

    for (int b = 0; b < batch; ++b) {
        P_LOOP_START(out, out_feature)
            for (int out_x = 0; out_x < filter->config.output_size; ++out_x) {

                float d_o = d_out[out_x * out + out_feature + batch * out_size];
                float *d_x = d_x_out_buffer + b * dx_out_one_size + out_feature * inp_size;
                int input_row_offset = out_x * filter->config.stride;

                for (int i = 0; i < in; ++i) {
                    int x_row_offset = i * filter->config.input_size +
                                       input_row_offset;
                    const float *row_ptr =
                            filter->training_data->input_transposed + x_row_offset;
                    int weights_offset = out_feature * in * k_size + i * k_size;
                    const float *weights_ptr = filter->weights->W + weights_offset;
                    //
                    float d_kernel [k_size];
                    op_vec_mul_sc(row_ptr, d_o, d_kernel, k_size);
                    op_vec_add(gradient->d_W + weights_offset, d_kernel, gradient->d_W + weights_offset, k_size);

                    float d_x_portion [k_size];
                    op_vec_mul_sc(weights_ptr, d_o, d_x_portion, k_size);
                    op_vec_add(d_x + x_row_offset, d_x_portion, d_x + x_row_offset, k_size);
                }
            }
        P_LOOP_END
        for (int o = 0; o < out; ++o) {
            op_vec_add(
                d_x_out_buffer + in * filter->config.input_size * o + b * dx_out_one_size,
                d_x_transposed + b * inp_size,
                d_x_transposed + b * inp_size,
                inp_size
            );
        }
        op_mat_transp(d_x_transposed, gradient->d_X + b * inp_size, filter->config.input_size, in);
    }
}


















