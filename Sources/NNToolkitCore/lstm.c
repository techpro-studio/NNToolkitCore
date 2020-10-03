//
//  File.c
//  
//
//  Created by Alex on 01.10.2020.
//

#include "lstm.h"
#include "recurrent_shared.h"
#include "stdlib.h"

LSTMConfig LSTMConfigCreate(int inputFeatureChannels, int outputFeatureChannels, bool v2, int batchSize, ActivationFunction* reccurrentActivation, ActivationFunction* activation){
    LSTMConfig config;
    config.inputFeatureChannels = inputFeatureChannels;
    config.outputFeatureChannels = outputFeatureChannels;
    config.v2 = v2;
    config.activation = activation;
    config.reccurrentActivation = reccurrentActivation;
    return config;
}

LSTMFilter* LSTMFilterCreate(LSTMConfig config){
    LSTMFilter * filter = malloc(sizeof(LSTMFilter));
    int out = config.outputFeatureChannels;
    filter->config = config;
    filter->state = malloc(out * 2 * sizeof(float));
    filter->output = filter->state + out;
    filter->weights = malloc(sizeof(LSTMWeights));
    int weightsBufferSize = 2;
    return filter;
}

void LSTMCellApply(LSTMFilter *filter, const float *input, float* output){

}

void LSTMFilterApply(LSTMFilter *filter, const float *input, float* output){

}

void LSTMFilterDestroy(LSTMFilter *filter) {

}
