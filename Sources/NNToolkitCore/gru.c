//
//  gru.c
//  audio_test
//
//  Created by Alex on 21.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "gru.h"
#include "operations.h"
#include "stdlib.h"
#include "string.h"

GRUConfig GRUConfigCreate(int input_feature_channels, int output_feature_channels, bool flip_output_gates, bool v2, bool returnSequences, int batchSize, ActivationFunction reccurrent_activation, ActivationFunction activation){
    GRUConfig config;
    config.input_feature_channels = input_feature_channels;
    config.timesteps = batchSize;
    config.v2 = v2;
    config.return_sequences = returnSequences;
    config.output_feature_channels = output_feature_channels;
    config.flip_output_gates = flip_output_gates;
    config.reccurrent_activation = reccurrent_activation;
    config.activation = activation;
    return config;
}

struct GRUStruct {
    GRUConfig config;
    float *buffer;
    float *state;
    GRUWeights* weights;
};

GRUWeights* GRUGetWeights(GRU filter){
    return filter->weights;
}


GRU GRUCreateForInference(GRUConfig config) {
    GRU filter = malloc(sizeof(struct GRUStruct));
    filter->config = config;
    filter->weights = malloc(sizeof(GRUWeights));
    int in = config.input_feature_channels;
    int out = config.output_feature_channels;
    int buff_state_length = 7 * out * sizeof(float);
    filter->state = malloc(buff_state_length);
    memset(filter->state, 0, buff_state_length);
    filter->buffer = filter->state + out;
    int length = (3 * in * out + 3 * out * out + 6 * out) * sizeof(float);
    float *weights = malloc(length);
    memset(weights, 0, length);
    filter->weights->W_z = weights;
    filter->weights->W_r = filter->weights->W_z + in * out;
    filter->weights->W_h = filter->weights->W_r + in * out;
    filter->weights->U_z = filter->weights->W_h + in * out;
    filter->weights->U_r = filter->weights->U_z + out * out;
    filter->weights->U_h = filter->weights->U_r + out * out;
    filter->weights->b_iz = filter->weights->U_h + out * out;
    filter->weights->b_hz = filter->weights->b_iz + out;
    filter->weights->b_ir = filter->weights->b_hz + out;
    filter->weights->b_hr = filter->weights->b_ir + out;
    filter->weights->b_ih = filter->weights->b_hr + out;
    filter->weights->b_hh = filter->weights->b_ih + out;
    return filter;
}


void GRUDestroy(GRU filter) {
    free(filter->weights->W_z);
    free(filter->state);
    free(filter->weights);
    free(filter);
}

void ComputeGate(int in, int out, ActivationFunction activation, const float *x, const float*h, const float *W, const float *U, const float* b_i, const float* b_h, bool useHiddenBias,  float* gate) {
    // out = x * W
    op_mat_mul(x, W, gate, 1, out, in, 0.0);
//    out = x * W + b_i
    op_vec_add(gate, b_i, gate, out);
    // in_U = h_t * U
    op_mat_mul(h, U, gate, 1, out, out, 1.0);
    // g = g + b;
    if (useHiddenBias){
        op_vec_add(gate, b_h, gate, out);
    }
    // g = activation(g);
    if (activation){
        ActivationFunctionApply(activation, gate, gate);
    }
}


void GRUCellCompute(GRU filter, const float *x, const float *h_pr, float* ht, float *buffer) {
    int out = filter->config.output_feature_channels;
    int in = filter->config.input_feature_channels;
    // z = sigmoid(x * W_z + h_pr * U_z + bz)
    float* z = buffer;
    ComputeGate(in, out, filter->config.reccurrent_activation, x, h_pr, filter->weights->W_z, filter->weights->U_z, filter->weights->b_iz, filter->weights->b_hz, filter->config.v2, z);
    // r = sigmoid(x * W_r + h_pr * U_r + br)
    float* r = z + out;
    ComputeGate(in, out, filter->config.reccurrent_activation, x, h_pr, filter->weights->W_r, filter->weights->U_r, filter->weights->b_ir, filter->weights->b_hr, filter->config.v2, r);
    //
    // h_tilda = tanh(x * W_h + b_ih +  r <*> (h_prev * U_h + b_ih));
    float* h_tilda = r + out;
    //x * W_h
    op_mat_mul(x, filter->weights->W_h, h_tilda, 1, out, in, 0.0);
    //x * W_h + b_ih
    op_vec_add(h_tilda, filter->weights->b_ih, h_tilda, out);
    
    float *h_prev_Uh = h_tilda + out;

    // V2
    if (filter->config.v2) {
//    h_prev * UH
        op_mat_mul(h_pr, filter->weights->U_h, h_prev_Uh, 1, out, out, 0.0);
    //    (h_prev * UH + b_hh)
        op_vec_add(h_prev_Uh, filter->weights->b_hh, h_prev_Uh, out);
        //(h_prev * UH + b_hh) <*>r
        op_vec_mul(r, h_prev_Uh, h_prev_Uh, out);
        //x * W_h + b_ih + h_prev_UH
        op_vec_add(h_tilda, h_prev_Uh, h_tilda, out);
    } else {
        // (hprev <*> r)
        op_vec_mul(r, h_pr, h_prev_Uh, out);
        // UH * (hprev <*> r) + h_tida
        op_mat_mul(h_prev_Uh, filter->weights->U_h, h_tilda, 1, out, out, 1.0);
    }

    //tanh(x * W_h + (h_prev <*> r) * U_h + bh);
    ActivationFunctionApply(filter->config.activation, h_tilda, h_tilda);
//    filter->config.activation(h_tilda, h_tilda, out);
    // h_t = (1 - z) <*> h_pr + z <*> h_tilda;
    // ht = -z;
    float * minus_z_pw = h_prev_Uh + out;
    op_vec_neg(z, minus_z_pw, out);
    //ht= -z + 1
    op_vec_add_sc(minus_z_pw,  1, minus_z_pw, out);
    //ht = (1 - z) <*> h_tilda ? h_pr flip?
    op_vec_mul(minus_z_pw, filter->config.flip_output_gates ? h_pr : h_tilda, minus_z_pw, out);
    //h_tilda = z <*> h_tild ? h_pr flip?
    float *z_h_pw = minus_z_pw + out;
    op_vec_mul(z, filter->config.flip_output_gates ? h_tilda : h_pr, z_h_pw, out);
    op_vec_add(minus_z_pw, z_h_pw, ht, out);
}

int GRUApplyInference(GRU filter, const float *input, float* output){
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


