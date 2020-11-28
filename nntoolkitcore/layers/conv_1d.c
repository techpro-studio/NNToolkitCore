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
#include "nntoolkitcore/core/memory.h"
#include "nntoolkitcore/layers/private/weights_private.h"

typedef struct {
    ConvTrainingConfig config;
    float *input_transposed;
    DefaultGradient **batch_gradients;
} Conv1dTrainingData;

typedef struct {
    void *buffer;
} Conv1dInferenceData;

struct Conv1dStruct {
    Conv1dConfig config;
    ConvWeights *weights;
    Conv1dInferenceData *inference_data;
    Conv1dTrainingData *training_data;
};

ConvWeightsSize conv1d_weight_size_from_config(Conv1dConfig config) {
    int w_size = config.kernel_size * config.input_feature_channels * config.output_feature_channels;
    int sum = w_size + config.output_feature_channels;
    return (DefaultWeightsSize) {.w = w_size, .b = config.output_feature_channels, .sum = sum};
}


static Conv1dInferenceData *conv1d_inference_data_create(Conv1dConfig config) {
    Conv1dInferenceData *data = malloc(sizeof(Conv1dInferenceData));
    data->buffer = malloc(config.input_size * config.input_feature_channels * sizeof(float));
    return data;
}

static Conv1dTrainingData *conv1d_training_data_create(Conv1dConfig config, ConvTrainingConfig training_config) {
    Conv1dTrainingData *data = malloc(sizeof(Conv1dTrainingData));
    int b = training_config.mini_batch_size;
    data->config = training_config;
    data->input_transposed = malloc(config.input_feature_channels * config.input_size
                                    * b * sizeof(float));
    data->batch_gradients = malloc(b * sizeof(DefaultGradient *));
    for (int i = 0; i < b; ++i) {
        data->batch_gradients[i] = default_gradient_create(conv1d_weight_size_from_config(config), 0);
    }
    return data;
}

static void conv_training_data_destroy(Conv1dTrainingData *training_data) {
    for (int i = 0; i < training_data->config.mini_batch_size; ++i) {
        default_gradient_destroy(training_data->batch_gradients[i]);
    }
    free(training_data->batch_gradients);
    free(training_data->input_transposed);
    free(training_data);
}


static void conv1d_inference_data_destroy(Conv1dInferenceData *data) {
    free(data->buffer);
    free(data);
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

Conv1d conv1d_create(Conv1dConfig config) {
    Conv1d filter = malloc(sizeof(struct Conv1dStruct));
    filter->config = config;
    filter->weights = default_weights_create(conv1d_weight_size_from_config(config));
    filter->training_data = NULL;
    filter->inference_data = NULL;
    return filter;
}

Conv1d Conv1dCreateForInference(Conv1dConfig config) {
    Conv1d filter = conv1d_create(config);
    filter->inference_data = conv1d_inference_data_create(config);
    return filter;
}

Conv1d Conv1dCreateForTraining(Conv1dConfig config, ConvTrainingConfig training_config) {
    Conv1d filter = conv1d_create(config);
    filter->training_data = conv1d_training_data_create(config, training_config);
    return filter;
}

void Conv1dDestroy(Conv1d filter) {
    if (filter->inference_data != NULL) {
        conv1d_inference_data_destroy(filter->inference_data);
    }
    if (filter->training_data != NULL) {
        conv_training_data_destroy(filter->training_data);
    }
    free(filter->weights->W);
    free(filter->weights);
    free(filter);
}

static void conv_1d_one(const struct Conv1dStruct *filter, const float *input, float *output,
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
    conv_1d_one(filter, input, output, filter->inference_data->buffer);
    return 0;
}

ConvGradient *Conv1dCreateGradient(Conv1dConfig config, ConvTrainingConfig training_config) {
    return default_gradient_create(conv1d_weight_size_from_config(config),
                                   training_config.mini_batch_size *
                                   config.input_size * config.input_feature_channels);
}

void ConvGradientDestroy(ConvGradient *gradient) {
    default_gradient_destroy(gradient);
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
    int db_size = filter->config.output_feature_channels *
                  filter->training_data->config.mini_batch_size;
    for (int o = 0; o < filter->config.output_size; ++o) {
        op_vec_add(gradient->d_b,d_out + o * db_size, gradient->d_b, db_size);
    }


    int k_size = filter->config.kernel_size;
    int batch = filter->training_data->config.mini_batch_size;
    int in_ftrs = filter->config.input_feature_channels;
    int out_ftrs = filter->config.output_feature_channels;
    int W_size = in_ftrs * out_ftrs * k_size;
    int inp_size = in_ftrs * filter->config.input_size;
    int out_size = out_ftrs * filter->config.output_size;
    int buffer_size = inp_size * batch;

    float *d_x_transposed = f_malloc(buffer_size);


    //then thought the h features
    //        out_f   f   f
    //  out_n  d1  d2  d3
    //  out_n  d4  d5  d6

    for (int b = 0; b < batch; ++b) {
        for (int out_f = 0; out_f < out_ftrs; ++out_f) {
            for (int out_n = 0; out_n < filter->config.output_size; ++out_n) {

                float d_o = d_out[out_n * out_ftrs + out_f + b * out_size];

                for (int in_f = 0; in_f < in_ftrs; ++in_f) {

                    int x_row_offset = in_f * filter->config.input_size +
                                       out_n * filter->config.stride;

                    const float *row_ptr =
                            filter->training_data->input_transposed + b * inp_size + x_row_offset;

                    int weights_offset = out_f * in_ftrs * k_size + in_f * k_size;
                    const float *weights_ptr = filter->weights->W + weights_offset;

                    // d_W

                    float d_kernel[k_size];
                    op_vec_mul_sc(row_ptr, d_o, d_kernel, k_size);
                    float *d_W = filter->training_data->batch_gradients[b]->d_W + weights_offset;
                    op_vec_add(d_W, d_kernel, d_W, k_size);

                    // d_X;

                    float *d_X = d_x_transposed + b * inp_size + x_row_offset;
                    float d_x_portion[k_size];
                    op_vec_mul_sc(weights_ptr, d_o, d_x_portion, k_size);
                    op_vec_add(d_X, d_x_portion, d_X, k_size);
                }
            }
        }
        op_mat_transp(d_x_transposed + b * inp_size, gradient->d_X + b * inp_size, filter->config.input_size, in_ftrs);
    }

}


















