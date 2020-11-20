//
// Created by Alex on 19.11.2020.
//

#include "nntoolkitcore/core/memory.h"
#include "nntoolkitcore/layers/private/recurrent_private.h"
#include "rnn.h"
#include "stdlib.h"
#include "nntoolkitcore/core/ops.h"


typedef struct {
    float *computation_buffer;
    float *gate;
    float *input;
    float *output;
} RNNTrainingData;

typedef struct {
    float *computation_buffer;
} RNNInferenceData;

typedef RecurrentWeightsSize RNNWeightsSize;

struct RNNStruct {
    RNNWeights *weights;
    RNNConfig config;
    float *h;
    RNNInferenceData *inference_data;
    RNNTrainingData* training_data;
};


RNNWeightsSize rnn_weights_size_from_config(RNNConfig config){
    int in = config.input_feature_channels;
    int out = config.output_feature_channels;
    RNNWeightsSize size;
    size.w = in * out;
    size.u = out * out;
    size.b_i = out;
    size.b_h = out;
    size.sum = size.w + size.u + size.b_h + size.b_i;
    return size;
}

RNNConfig RNNConfigCreate(
    int input_feature_channels,
    int output_feature_channels,
    bool v2,
    bool return_sequences,
    int timesteps,
    ActivationFunction activation
) {
    RNNConfig result;
    result.input_feature_channels = input_feature_channels;
    result.output_feature_channels = output_feature_channels;
    result.v2 = v2;
    result.return_sequences = return_sequences;
    result.timesteps = timesteps;
    result.activation = activation;
    return result;
}

void rnn_training_data_destroy(RNNTrainingData *data){
    free(data->computation_buffer);
    free(data);
}

void rnn_inference_data_destroy(RNNInferenceData *data){
    free(data->computation_buffer);
    free(data);
}

RNNInferenceData *rnn_inference_data_create(RNNConfig config){
    RNNInferenceData *data = malloc(sizeof(RNNInferenceData));
    data->computation_buffer = f_malloc(3 * config.output_feature_channels);
    return data;
}

RNNTrainingData *rnn_training_data_create(RNNConfig config, RecurrentTrainingConfig training_config){
    RNNTrainingData *data = malloc(sizeof(RNNTrainingData));
    int out = config.output_feature_channels;
    int batch = training_config.mini_batch_size;
    int input_size = batch * config.input_feature_channels * config.timesteps;
    int output_size = batch * out * config.timesteps;
    int gate_size = batch * out * config.timesteps;

    data->input = f_malloc(input_size + output_size + gate_size + 2 * out);
    data->output = data->input + input_size;
    data->gate = data->output + output_size;
    data->computation_buffer = data->gate + gate_size;
    return data;
}

RecurrentWeights *RNNGetWeights(RNN filter) {
    return filter->weights;
}

RNN rnn_create(RNNConfig config){
    RNN filter = malloc(sizeof(struct RNNStruct));
    filter->config = config;
    filter->weights = recurrent_weights_create(rnn_weights_size_from_config(config));
    filter->training_data = NULL;
    filter->inference_data = NULL;
    return filter;
}

RNN RNNCreateForInference(RNNConfig config) {
    RNN rnn = rnn_create(config);
    rnn->inference_data = rnn_inference_data_create(config);
    return rnn;
}

RNN RNNCreateForTraining(RNNConfig config, RecurrentTrainingConfig training_config) {
    RNN rnn = rnn_create(config);
    rnn->training_data = rnn_training_data_create(config, training_config);
    return rnn;
}

RNNGradient *RNNGradientCreate(RNNConfig config, RNNTrainingConfig training_config) {
    return recurrent_gradient_create(
            rnn_weights_size_from_config(config),
            training_config,
            config.timesteps * config.input_feature_channels
    );
}

void RNNDestroy(RNN filter) {
    recurrent_weights_destroy(filter->weights);
    free(filter->h);
    if (filter->training_data != NULL){
        rnn_training_data_destroy(filter->training_data);
    }
    if (filter->inference_data != NULL){
        rnn_inference_data_destroy(filter->inference_data);
    }
}

void RNNCellForward(
    RecurrentWeights *weights,
    ActivationFunction activation,
    bool v2,
    int in,
    int out,
    const float *input,
    float *h_prev,
    float *h,
    float *buffer,
    float *gate
){
    float *x_W = buffer;
    op_mat_mul(input, weights->W, x_W, 1, out, in);
    op_vec_add(x_W, weights->b_i, x_W, out);
    float *h_U = x_W + out;
    op_mat_mul(h_prev, weights->U, h_U, 1, out, out);
    if (v2){
        op_vec_add(h_U, weights->b_h, h_U, out);
    }
    op_vec_add(h_U, x_W, gate, out);
    ActivationFunctionApply(activation, gate, h);
}

int RNNApplyInference(RNN filter, const float *input, float *output) {
    if(filter->training_data != NULL){
        return -1;
    }
    int out = filter->config.output_feature_channels;
    int in = filter->config.input_feature_channels;
    for (int i = 0; i < filter->config.timesteps; ++i) {
        int output_offset = filter->config.return_sequences ? i * out : 0;
        RNNCellForward(
            filter->weights,
            filter->config.activation,
            filter->config.v2,
            in, out, input + i * in,
            filter->h,
            output + i * output_offset,
            filter->inference_data->computation_buffer,
            filter->inference_data->computation_buffer + 2 * out
        );
        f_copy(filter->h, output + output_offset, out);
    }
    return 0;
}

int RNNApplyTrainingBatch(RNN filter, const float *input, float *output) {
    if (filter->training_data == NULL){
        return -1;
    }
    int out = filter->config.output_feature_channels;
    int in = filter->config.input_feature_channels;
    return 0;
}

void RNNCalculateGradient(RNN filter, RecurrentGradient *gradients, float *d_out) {

}




