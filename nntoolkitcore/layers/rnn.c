//
// Created by Alex on 19.11.2020.
//

#include "rnn.h"
#include "stdlib.h"
#include "nntoolkitcore/core/ops.h"

RNNGradient * RNNGradientCreate(RNNConfig config, RNNTrainingConfig training_config) {
    return NULL;
}

RNNConfig
RNNConfigCreate(
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
    return NULL;
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

RNN RNNCreateForInference(RNNConfig config) {
    return NULL;
}

RNN RNNCreateForTraining(RNNConfig config, RNNTrainingConfig training_config) {
    return NULL;
}

int RNNApplyInference(RNN filter, const float *input, float *output) {
    return 0;
}

int RNNApplyTrainingBatch(RNN filter, const float *input, float *output) {
    return 0;
}

void RNNCalculateGradient(RNN filter, RNNGradient *gradients, float *d_out) {

}

void RNNDestroy(RNN filter) {

}
