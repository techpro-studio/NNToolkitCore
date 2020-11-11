//
// Created by Alex on 05.11.2020.
//

#include "nntoolkitcore/layers/gru_2.h"
#include "nntoolkitcore/core/ops.h"
#include "stdlib.h"
#include "string.h"

GRU2Config GRU2ConfigCreate(int input_feature_channels, int output_feature_channels,  bool return_sequences, int batchSize, ActivationFunction recurrent_activation, ActivationFunction activation){
    GRU2Config config;
    config.input_feature_channels = input_feature_channels;
    config.timesteps = batchSize;
    config.return_sequences = return_sequences;
    config.output_feature_channels = output_feature_channels;
    config.recurrent_activation = recurrent_activation;
    config.activation = activation;
    return config;
}

struct GRU2Struct {
    GRU2Config config;
    float *buffer;
    float *state;
    GRU2Weights * weights;
};

GRU2Weights* GRU2GetWeights(GRU2 filter){
    return filter->weights;
}

GRU2 GRU2CreateForInference(GRU2Config config) {
    GRU2 filter = malloc(sizeof(struct GRU2Struct));
    filter->config = config;
    filter->weights = malloc(sizeof(GRU2Weights));
    int in = config.input_feature_channels;
    int out = config.output_feature_channels;
    int buff_state_length = 14 * out * sizeof(float);
    filter->state = malloc(buff_state_length);
    memset(filter->state, 0, buff_state_length);
    filter->buffer = filter->state + out;
    int length = (3 * in * out + 3 * out * out + 6 * out) * sizeof(float);
    float *weights = malloc(length);
    memset(weights, 0, length);
    filter->weights->W = weights;
    filter->weights->U = filter->weights->W + 3 * in * out;
    filter->weights->b_i = filter->weights->U + 3 * out * out;
    filter->weights->b_h = filter->weights->b_i + 3 * out;
    return filter;
}


void GRU2Destroy(GRU2 filter) {
    free(filter->weights->W);
    free(filter->state);
    free(filter->weights);
    free(filter);
}


static void GRUCellCompute(GRU2 filter, const float *x, const float *h_pr, float* ht, float *buffer) {
    int out = filter->config.output_feature_channels;
    int in = filter->config.input_feature_channels;
    // W = [Wz, Wr, Wh]
    // U = [Uz, Ur, Uh]
    // x_W = x * W
    float* x_W = buffer;
    op_mat_mul(x, filter->weights->W, x_W, 1, 3 * out, in);
    // b = [bz, br, bh]
    // x_W += bi
    op_vec_add(x_W, filter->weights->b_i, x_W, 3 * out);
    float* h_pr_U = x_W + 3 * out;
    op_mat_mul(h_pr, filter->weights->U, h_pr_U, 1, 3 * out, out);
    op_vec_add(h_pr_U, filter->weights->b_h, h_pr_U, 3 * out);
    // Z_zr = x_W[0: 2 * out] + h_pr_U[0: 2 * out]
    float* Z_zr = h_pr_U + 3 * out;
    op_vec_add(x_W, h_pr_U, Z_zr, 2 * out);
    // z = recurrent_activation(Z_zr[0: out])
    // r = recurrent_activation(Z_zr[out: 2 * out])
    float *z = Z_zr + 2 * out;
    float *r = z + out;
    ActivationFunctionApply(filter->config.recurrent_activation, Z_zr, z);
    ActivationFunctionApply(filter->config.recurrent_activation, Z_zr + out, r);
    //x * W_h
    float *h_tilda = r + out;
    //h_pr_U[2* out : 3 * out] <*> r
    op_vec_mul(r, h_pr_U + 2 * out, h_tilda, out);
    //h_tilda += x_W[2 * out: 3 * out]
    op_vec_add(h_tilda, x_W + 2 * out, h_tilda, out);
    //tanh(a_H);
    ActivationFunctionApply(filter->config.activation, h_tilda, h_tilda);
//    filter->config.activation(h_tilda, h_tilda, out);
    // h_t = (1 - z) <*> h_pr + z <*> h_tilda;
    // ht = -z;
    float * minus_z_pw = h_tilda + out;
    op_vec_neg(z, minus_z_pw, out);
    //ht= -z + 1
    op_vec_add_sc(minus_z_pw,  1, minus_z_pw, out);
    //ht = (1 - z) <*> h_tilda
    op_vec_mul(minus_z_pw, h_tilda, minus_z_pw, out);
    //h_tilda = z <*> h_pr
    float *z_h_pw = minus_z_pw + out;
    op_vec_mul(z, h_pr, z_h_pw, out);
    op_vec_add(minus_z_pw, z_h_pw, ht, out);
}

int GRU2ApplyInference(GRU2 filter, const float *input, float* output){
//    if(filter->training_data != NULL){
//        return -1;
//    }
    int out = filter->config.output_feature_channels;
    int in = filter->config.input_feature_channels;
    for (int i = 0; i < filter->config.timesteps; ++i){
        int output_offset = filter->config.return_sequences ? i * out : 0;
        GRUCellCompute(filter, input + i * in, filter->state, output + output_offset, filter->buffer);
        memcpy(filter->state, output + output_offset, out * sizeof(float));
    }
    return 0;
}


