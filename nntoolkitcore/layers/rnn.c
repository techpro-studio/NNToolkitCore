//
// Created by Alex on 19.11.2020.
//

#include <nntoolkitcore/core/memory.h>
#include "rnn.h"
#include "stdlib.h"
#include "nntoolkitcore/core/ops.h"


typedef struct {
    float *computation_buffer;
    float *gate;
} RNNTrainingData;

typedef struct {
    float *computation_buffer;
} RNNInferenceData;

typedef RecurrentWeightsSize RNNWeightsSize;

struct RNNStruct {
    RecurrentWeights *weights;
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
    size.buffer = (size.w + size.u + size.b_h + size.b_i) * sizeof(float);
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
    data->computation_buffer = malloc_zeros(3 * config.output_feature_channels * sizeof(float));
    return data;
}

RNNTrainingData *rnn_training_data_create(RNNConfig config, RecurrentTrainingConfig training_config){
    RNNTrainingData *data = malloc(sizeof(RNNTrainingData));
    int out = config.output_feature_channels;
    data->computation_buffer = malloc_zeros(3 * out * sizeof(float));
    data->gate = data->computation_buffer + 2 * out;
    return data;
}

RecurrentWeights *RNNGetWeights(RNN filter) {
    return filter->weights;
}

RNN rnn_create(RNNConfig config){
    RNN filter = malloc(sizeof(struct RNNStruct));
    filter->config = config;
    filter->weights = malloc(sizeof(RecurrentWeights));

    filter->training_data = NULL;
    filter->inference_data = NULL;

    RecurrentWeightsSize w_sizes = rnn_weights_size_from_config(config);

    filter->weights->W = malloc_zeros(w_sizes.buffer);
    filter->weights->U = filter->weights->W + w_sizes.w;
    filter->weights->b_i = filter->weights->U + w_sizes.u;
    filter->weights->b_h = filter->weights->b_i + w_sizes.b_i;

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
    return RecurrentGradientCreate(
            rnn_weights_size_from_config(config),
            training_config,
            config.timesteps * config.input_feature_channels
    );
}

void RNNDestroy(RNN filter) {
    free(filter->weights->W);
    free(filter->weights);
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
        memcpy(filter->h, output + output_offset, out * sizeof(float));
    }
    return 0;
}

int RNNApplyTrainingBatch(RNN filter, const float *input, float *output) {
    return 0;
}

void RNNCalculateGradient(RNN filter, RecurrentGradient *gradients, float *d_out) {

}




