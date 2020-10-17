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

GRUConfig GRUConfigCreate(int input, int output, bool flipOutputGates, bool v2, bool returnSequences, int batchSize, ActivationFunction* reccurrent_activation, ActivationFunction* activation){
    GRUConfig config;
    config.inputFeatureChannels = input;
    config.timesteps = batchSize;
    config.v2 = v2;
    config.returnSequences = returnSequences;
    config.outputFeatureChannels = output;
    config.flipOutputGates = flipOutputGates;
    config.reccurrentActivation = reccurrent_activation;
    config.activation = activation;
    return config;
}

struct GRUFilterStruct {
    GRUConfig config;
    float *buffer;
    float *state;
    GRUWeights* weights;
} ;

GRUFilter GRUFilterCreate(GRUConfig config) {
    GRUFilter filter = malloc(sizeof(struct GRUFilterStruct));
    filter->config = config;
    filter->weights = malloc(sizeof(GRUWeights));
    int in = config.inputFeatureChannels;
    int out = config.outputFeatureChannels;
    int buffStateLength = 7 * out * sizeof(float);
    filter->state = malloc(buffStateLength);
    memset(filter->state, 0, buffStateLength);
    filter->buffer = filter->state + out;
    int length = (3 * in * out + 3 * out * out + 6 * out) * sizeof(float);
    float *weights = malloc(length);
    memset(weights, 0, length);
    filter->weights->Wz = weights;
    filter->weights->Wr = filter->weights->Wz + in * out;
    filter->weights->Wh = filter->weights->Wr + in * out;
    filter->weights->Uz = filter->weights->Wh + in * out;
    filter->weights->Ur = filter->weights->Uz + out * out;
    filter->weights->Uh = filter->weights->Ur + out * out;
    filter->weights->b_iz = filter->weights->Uh + out * out;
    filter->weights->b_hz = filter->weights->b_iz + out;
    filter->weights->b_ir = filter->weights->b_hz + out;
    filter->weights->b_hr = filter->weights->b_ir + out;
    filter->weights->b_ih = filter->weights->b_hr + out;
    filter->weights->b_hh = filter->weights->b_ih + out;
    return filter;
}


void GRUFilterDestroy(GRUFilter filter) {
    free(filter->weights->Wz);
    free(filter->state);
    free(filter->weights);
    free(filter);
}

void ComputeGate(int in, int out, ActivationFunction* activation, const float *x, const float*h, const float *W, const float *U, const float* b_i, const float* b_h, bool useHiddenBias,  float* gate) {
    // out = x * W
    MatMul(x, W, gate, 1, out, in, 0.0);
//    out = x * W + b_i
    VectorAdd(gate, b_i, gate, out);
    // in_U = h_t * U
    MatMul(h, U, gate, 1, out, out, 1.0);
    // g = g + b;
    if (useHiddenBias){
        VectorAdd(gate, b_h, gate, out);
    }
    // g = activation(g);
    if (activation){
        ActivationFunctionApply(activation, gate, gate);
    }
}


void GRUCellCompute(GRUFilter filter, const float *x, const float *h_pr, float* ht, float *buffer) {
    int out = filter->config.outputFeatureChannels;
    int in = filter->config.inputFeatureChannels;
    // z = sigmoid(x * Wz + h_pr * Uz + bz)
    float* z = buffer;
    ComputeGate(in, out, filter->config.reccurrentActivation,  x, h_pr, filter->weights->Wz, filter->weights->Uz, filter->weights->b_iz, filter->weights->b_hz, filter->config.v2,  z);
    // r = sigmoid(x * Wr + h_pr * Ur + br)
    float* r = z + out;
    ComputeGate(in, out, filter->config.reccurrentActivation, x, h_pr, filter->weights->Wr, filter->weights->Ur, filter->weights->b_ir, filter->weights->b_hr, filter->config.v2, r);
    //
    // h_tilda = tanh(x * Wh + b_ih +  r <*> (h_prev * Uh + b_ih));
    float* h_tilda = r + out;
    //x * Wh
    MatMul(x, filter->weights->Wh, h_tilda, 1, out, in, 0.0);
    //x * Wh + b_ih
    VectorAdd(h_tilda, filter->weights->b_ih, h_tilda, out);
    
    float *h_prev_Uh = h_tilda + out;

    // V2
    if (filter->config.v2) {
//    h_prev * UH
        MatMul(h_pr, filter->weights->Uh, h_prev_Uh, 1, out, out, 0.0);
    //    (h_prev * UH + b_hh)
        VectorAdd(h_prev_Uh, filter->weights->b_hh, h_prev_Uh, out);
        //(h_prev * UH + b_hh) <*>r
        VectorMul(r, h_prev_Uh, h_prev_Uh, out);
        //x * Wh + b_ih + h_prev_UH
        VectorAdd(h_tilda, h_prev_Uh, h_tilda, out);
    } else {
        // (hprev <*> r)
        VectorMul(r, h_pr, h_prev_Uh, out);
        // UH * (hprev <*> r) + h_tida
        MatMul(h_prev_Uh, filter->weights->Uh, h_tilda, 1, out, out, 1.0);
    }

    //tanh(x * Wh + (h_prev <*> r) * Uh + bh);
    ActivationFunctionApply(filter->config.activation, h_tilda, h_tilda);
//    filter->config.activation(h_tilda, h_tilda, out);
    // h_t = (1 - z) <*> h_pr + z <*> h_tilda;
    // ht = -z;
    float * minus_z_pw = h_prev_Uh + out;
    VectorNeg(z, minus_z_pw, out);
    //ht= -z + 1
    VectorAddS(minus_z_pw, 1, minus_z_pw, out);
    //ht = (1 - z) <*> h_tilda ? h_pr flip?
    VectorMul(minus_z_pw, filter->config.flipOutputGates ? h_pr : h_tilda, minus_z_pw, out);
    //h_tilda = z <*> h_tild ? h_pr flip?
    float *z_h_pw = minus_z_pw + out;
    VectorMul(z, filter->config.flipOutputGates ? h_tilda : h_pr, z_h_pw, out);
    VectorAdd(minus_z_pw, z_h_pw, ht, out);
}

void GRUFilterApply(GRUFilter filter, const float *input, float* output){
    int out = filter->config.outputFeatureChannels;
    int in = filter->config.inputFeatureChannels;
    for (int i = 0; i < filter->config.timesteps; ++i){
        int outputIndex = filter->config.returnSequences ? i * out : 0;
        GRUCellCompute(filter, input + i * in, filter->state, output + outputIndex, filter->buffer);
        memcpy(filter->state, output + outputIndex, out * sizeof(float));
    }
}


