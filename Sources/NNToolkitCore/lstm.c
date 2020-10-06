//
//  File.c
//  
//
//  Created by Alex on 01.10.2020.
//

#include "lstm.h"
#include "stdlib.h"
#include "string.h"
#include "activation.h"
#include "operations.h"

LSTMConfig LSTMConfigCreate(int inputFeatureChannels, int outputFeatureChannels, bool v2, bool returnSequences, int batchSize, ActivationFunction* reccurrentActivation, ActivationFunction* activation){
    LSTMConfig config;
    config.inputFeatureChannels = inputFeatureChannels;
    config.outputFeatureChannels = outputFeatureChannels;
    config.v2 = v2;
    config.returnSequences = returnSequences;
    config.activation = activation;
    config.batchSize = batchSize;
    config.reccurrentActivation = reccurrentActivation;
    return config;
}

LSTMFilter* LSTMFilterCreate(LSTMConfig config){
    LSTMFilter * filter = malloc(sizeof(LSTMFilter));
    int out = config.outputFeatureChannels;
    filter->config = config;
    int h_and_c_size = out * 2 * sizeof(float);
    filter->state = malloc(h_and_c_size);
    filter->output = filter->state + out;
    memset(filter->state, 0, h_and_c_size);
    filter->weights = malloc(sizeof(LSTMWeights));
    int in = config.inputFeatureChannels;
    int computationBufferSize = 7 * out * sizeof(float);
    filter->buffer = malloc(computationBufferSize);
    memset(filter->buffer, 0, computationBufferSize);
    int weightsBufferSize = 4 * in * out + 4 * out * out + 2 * 4 * out * sizeof(float);
    float *buffer = malloc(weightsBufferSize);
    memset(buffer, 0, weightsBufferSize);
    filter->weights->W = buffer;
    filter->weights->U = filter->weights->W + 4 * in * out;
    filter->weights->b_i = filter->weights->U + 4 * out * out;
    filter->weights->b_h = filter->weights->b_i + 4 * out;
    return filter;
}

void LSTMFilterDestroy(LSTMFilter *filter) {
    free(filter->weights->W);
    free(filter->buffer);
    free(filter->weights);
    free(filter->state);
    free(filter);
}

void LSTMCellCompute(LSTMFilter *filter, const float *input, const float *c_pr, const float *h_pr, float *c, float* h, float * buffer){
    int out = filter->config.outputFeatureChannels;
    int in  = filter->config.inputFeatureChannels;
    float *Z = buffer;
    // Z = input * W (WI, WF, WG, WO in row) + h * U (UI, UF, UG, UO in row) + bias(BI, BF, BG, BO)
    MatMul(input, filter->weights->W, Z, 1, 4 * out, in, 0.0);
//    out = x * W + b_i
    VectorAdd(Z, filter->weights->b_i, Z, 4 * out);
    // in_U = h_t * U
    MatMul(h_pr, filter->weights->U, Z, 1, out * 4, out, 1.0);

    // input Gate =  recurrent_activation(Z[0: out])
    float* i = Z;
    ActivationFunctionApply(filter->config.reccurrentActivation, i, i);
    // forget Gate =  recurrent_activation(Z[out: 2 * out])
    float* f = i + out;
    ActivationFunctionApply(filter->config.reccurrentActivation, f, f);
    // candidate Gate = activation(Z[2 * out: 3 *out])
    float *g = f + out;
    ActivationFunctionApply(filter->config.activation, g, g);
    // output Gate = recurrent_activation(Z[3 * out: 4 *out])
    float *o = g + out;
    ActivationFunctionApply(filter->config.reccurrentActivation, o, o);
    // ig = i * g
    float* i_g = o + out;
    VectorMul(i, g, i_g, out);
    //f_cpr = f * c_pr
    float*  f_c_pr = i_g + out;
    VectorMul(f, c_pr, f_c_pr, out);
    //c = fc_pr + i_g
    VectorAdd(f_c_pr, i_g, c, out);
    // H = o * tanh(c)
    float* c_tanh = f_c_pr + out;
    ActivationFunctionApply(filter->config.activation, c, c_tanh);
    VectorMul(o, c_tanh, h, out);
}

void LSTMFilterApply(LSTMFilter *filter, const float *input, float* output){
    int out = filter->config.outputFeatureChannels;
    int in = filter->config.inputFeatureChannels;
    for (int i = 0; i < filter->config.batchSize; ++i){
        float state[out];
        int outputIndex = filter->config.returnSequences ? i * out : 0;
        LSTMCellCompute(filter, input + i * in, filter->state, filter->output, state, output + outputIndex, filter->buffer);
        memcpy(filter->output, output + outputIndex, out * sizeof(float));
        memcpy(filter->state, state, out * sizeof(float));
    }
}


