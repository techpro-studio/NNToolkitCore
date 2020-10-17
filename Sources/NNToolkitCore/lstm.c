//
//  File.c
//  
//
//  Created by Alex on 01.10.2020.
//

#include "lstm.h"
#include "stdlib.h"
#include "string.h"
#include "activation_default.h"
#include "operations.h"


LSTMActivations LSTMActivationsCreate(ActivationFunction inputGateActivation, ActivationFunction forgetGateActivation, ActivationFunction candidateGateActivation, ActivationFunction outputGateActivation, ActivationFunction outputActivation){
    LSTMActivations activations;
    activations.candidateGateActivation = candidateGateActivation;
    activations.forgetGateActivation = forgetGateActivation;
    activations.inputGateActivation = inputGateActivation;
    activations.outputActivation = outputActivation;
    activations.outputGateActivation = outputGateActivation;
    return activations;
}

struct LSTMFilterStruct {
    LSTMConfig config;
    float *buffer;
    float *state;
    float *output;
    int miniBatchSize;
    LSTMWeights* weights;
};

LSTMActivations LSTMActivationsCreateDefault(int size){
    return LSTMActivationsCreate(
        ActivationFunctionCreateSigmoid(size),
        ActivationFunctionCreateSigmoid(size),
        ActivationFunctionCreateTanh(size),
        ActivationFunctionCreateSigmoid(size),
        ActivationFunctionCreateTanh(size)
   );
}

void LSTMActivationsDestroy(LSTMActivations activations){
    ActivationFunctionDestroy(activations.inputGateActivation);
    ActivationFunctionDestroy(activations.forgetGateActivation);
    ActivationFunctionDestroy(activations.candidateGateActivation);
    ActivationFunctionDestroy(activations.outputGateActivation);
    ActivationFunctionDestroy(activations.outputActivation);
}


LSTMWeights* LSTMFilterGetWeights(LSTMFilter filter){
    return filter->weights;
}

LSTMConfig LSTMConfigCreate(int inputFeatureChannels, int outputFeatureChannels, bool v2, bool returnSequences, int timesteps, LSTMActivations activations){
    LSTMConfig config;
    config.inputFeatureChannels = inputFeatureChannels;
    config.outputFeatureChannels = outputFeatureChannels;
    config.v2 = v2;
    config.returnSequences = returnSequences;
    config.activations = activations;
    config.timesteps = timesteps;
    return config;
}


LSTMFilter LSTMFilterCreateForTraining(LSTMConfig config, int miniBatchSize){
    LSTMFilter filter = malloc(sizeof(struct LSTMFilterStruct));

    return filter;
}

LSTMFilter LSTMFilterCreateForInference(LSTMConfig config){
    LSTMFilter filter = malloc(sizeof(struct LSTMFilterStruct));

    filter->miniBatchSize = 1;

    int out = config.outputFeatureChannels;
    filter->config = config;

    int h_and_c_size = out * 2 * sizeof(float);
    filter->state = malloc(h_and_c_size);
    filter->output = filter->state + out;
    memset(filter->state, 0, h_and_c_size);

    int in = config.inputFeatureChannels;
    int computationBufferSize = 7 * out * sizeof(float);
    filter->buffer = malloc(computationBufferSize);
    memset(filter->buffer, 0, computationBufferSize);

    filter->weights = malloc(sizeof(LSTMWeights));
    int weightsBufferSize = (4 * in * out + 4 * out * out + 2 * 4 * out) * sizeof(float);
    float *buffer = malloc(weightsBufferSize);
    memset(buffer, 0, weightsBufferSize);
    filter->weights->W = buffer;
    filter->weights->U = filter->weights->W + 4 * in * out;
    filter->weights->b_i = filter->weights->U + 4 * out * out;
    filter->weights->b_h = filter->weights->b_i + 4 * out;

    return filter;
}

void LSTMFilterDestroy(LSTMFilter filter) {
    free(filter->weights->W);
    free(filter->buffer);
    free(filter->weights);
    free(filter->state);
    free(filter);
}

void LSTMCellForward(LSTMWeights *weights, LSTMActivations activations, int in, int out, const float *input, float *c_prev, float* h_prev, float *c, float* h, float* buffer){
    float *Z = buffer;
    // Z = input * W (WI, WF, WG, WO in row) + h * U (UI, UF, UG, UO in row) + bias(BI, BF, BG, BO)
    MatMul(input, weights->W, Z, 1, 4 * out, in, 0.0);
//    out = x * W + b_i
    VectorAdd(Z, weights->b_i, Z, 4 * out);
    // in_U = h_t * U
    MatMul(h_prev, weights->U, Z, 1, out * 4, out, 1.0);
    // input Gate =  recurrent_activation(Z[0: out])
    float* i = Z;
    ActivationFunctionApply(activations.inputGateActivation, i, i);
    // forget Gate =  recurrent_activation(Z[out: 2 * out])
    float* f = i + out;
    ActivationFunctionApply(activations.forgetGateActivation, f, f);
    // candidate Gate = activation(Z[2 * out: 3 *out])
    float *g = f + out;
    ActivationFunctionApply(activations.candidateGateActivation, g, g);
    // output Gate = recurrent_activation(Z[3 * out: 4 *out])
    float *o = g + out;
    ActivationFunctionApply(activations.outputGateActivation, o, o);
    // ig = i * g
    float* i_g = o + out;
    VectorMul(i, g, i_g, out);
    //f_cpr = f * c_pr
    float*  f_c_pr = i_g + out;
    VectorMul(f, c_prev, f_c_pr, out);
    //c = fc_pr + i_g
    VectorAdd(f_c_pr, i_g, c, out);
    // H = o * tanh(c)
    float* c_tanh = f_c_pr + out;
    ActivationFunctionApply(activations.outputActivation, c, c_tanh);
    VectorMul(o, c_tanh, h, out);
}

void LSTMFilterApply(LSTMFilter filter, const float *input, float* output){
    int out = filter->config.outputFeatureChannels;
    int in = filter->config.inputFeatureChannels;
    for (int i = 0; i < filter->config.timesteps; ++i){
        float state[out];
        int outputIndex = filter->config.returnSequences ? i * out : 0;
        LSTMCellForward(filter->weights, filter->config.activations, in, out, input + i * in, filter->state, filter->output, state, output + outputIndex, filter->buffer);
        memcpy(filter->output, output + outputIndex, out * sizeof(float));
        memcpy(filter->state, state, out * sizeof(float));
    }
}


