//
// Created by Alex on 19.11.2020.
//

#include <nntoolkitcore/core/memory.h>
#include "rnn.h"
#include "stdlib.h"
#include "nntoolkitcore/core/ops.h"


typedef struct {
    float *computation_buffer;
} RNNTrainingData;

typedef struct {
    float *computation_buffer;
} RNNInferenceData;


struct RNNStruct {
    RNNWeights *weights;
    RNNConfig config;
    float *h;
    RNNInferenceData *inference_data;
    RNNTrainingData* training_data;
};

typedef struct {
    int w;
    int u;
    int b_i;
    int b_h;
    int buffer;
} RNNWeightsSize;

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

RNNWeights *RNNGetWeights(RNN filter) {
    return filter->weights;
}


void RNNCellForward(
    RNNWeights *weights,
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


RNN rnn_create(RNNConfig config){
    RNN filter = malloc(sizeof(struct RNNStruct));
    filter->config = config;
    filter->weights = malloc(sizeof(RNNWeights));
    filter->training_data = NULL;

    RNNWeightsSize w_sizes = rnn_weights_size_from_config(config);

    filter->weights->W = malloc_zeros(w_sizes.buffer);
    filter->weights->U = filter->weights->W + w_sizes.w;
    filter->weights->b_i = filter->weights->U + w_sizes.u;
    filter->weights->b_h = filter->weights->b_i + w_sizes.b_i;

    return filter;
}

RNN RNNCreateForInference(RNNConfig config) {

}

int RNNApplyInference(RNN filter, const float *input, float *output) {
    return 0;
}


RNN RNNCreateForTraining(RNNConfig config, RNNTrainingConfig training_config) {
    return NULL;
}

int RNNApplyTrainingBatch(RNN filter, const float *input, float *output) {
    return 0;
}

RNNGradient * RNNGradientCreate(RNNConfig config, RNNTrainingConfig training_config) {
    return NULL;
}

void RNNCalculateGradient(RNN filter, RNNGradient *gradients, float *d_out) {

}

void RNNDestroy(RNN filter) {

}
