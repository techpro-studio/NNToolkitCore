//
//  Header.h
//  
//
//  Created by Alex on 01.10.2020.
//

#ifndef lstm_h
#define lstm_h

#include <stdio.h>
#include "stdbool.h"
#import "activation.h"


#if defined __cplusplus
extern "C" {
#endif



typedef struct {
    int inputFeatureChannels;

    int outputFeatureChannels;

    bool v2;

    int batchSize;

    ActivationFunction* reccurrentActivation;

    ActivationFunction* activation;
} LSTMConfig;

LSTMConfig LSTMConfigCreate(int inputFeatureChannels, int outputFeatureChannels, bool v2, int batchSize, ActivationFunction* reccurrentActivation, ActivationFunction* activation);


typedef struct {
    float *W;
    float *U;
    float *b_i;
    float *b_h;
} LSTMWeights;


typedef struct {
    LSTMConfig config;
    float *buffer;
    float *state;
    float *output;
    LSTMWeights* weights;
} LSTMFilter;


LSTMFilter* LSTMFilterCreate(LSTMConfig config);

//feature channels in row, row major pointer

void LSTMFilterApply(LSTMFilter *filter, const float *input, float* output);

void LSTMFilterDestroy(LSTMFilter *filter);


#if defined __cplusplus
}
#endif


#endif /* Header_h */
