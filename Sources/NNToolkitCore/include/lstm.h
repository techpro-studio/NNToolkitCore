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
    ActivationFunction candidateGateActivation;
    ActivationFunction inputGateActivation;
    ActivationFunction forgetGateActivation;
    ActivationFunction outputGateActivation;
    ActivationFunction outputActivation;
} LSTMActivations;

LSTMActivations LSTMActivationsCreate(ActivationFunction inputGateActivation, ActivationFunction forgetGateActivation, ActivationFunction candidateGateActivation, ActivationFunction outputGateActivation, ActivationFunction outputActivation);

LSTMActivations LSTMActivationsCreateDefault(int size);

void LSTMActivationsDestroy(LSTMActivations activations);

typedef struct {
    int inputFeatureChannels;

    int outputFeatureChannels;

    bool v2;

    bool returnSequences;

    int timesteps;

    LSTMActivations activations;

} LSTMConfig;

LSTMConfig LSTMConfigCreate(int inputFeatureChannels, int outputFeatureChannels, bool v2, bool returnSequences, int timesteps, LSTMActivations lstmActivations);

typedef struct {
    float *W;
    float *U;
    float *b_i;
    float *b_h;
} LSTMWeights;

typedef struct {
    int mini_batch_size;
} LSTMTrainingConfig;

LSTMTrainingConfig LSTMTrainingConfigCreate(int miniBatchSize);

typedef struct {
    float * d_W;
    float * d_U;
    float * d_b_i;
    float * d_b_h;
    float * d_X;
} LSTMGradients;

typedef struct LSTMFilterStruct* LSTMFilter;

LSTMWeights* LSTMFilterGetWeights(LSTMFilter filter);

LSTMFilter LSTMFilterCreateForInference(LSTMConfig config);

LSTMFilter LSTMFilterCreateForTraining(LSTMConfig config, LSTMTrainingConfig training_config);

//feature channels in row, row major pointer

int LSTMFilterApply(LSTMFilter filter, const float *input, float* output);

int LSTMFilterApplyTrainingBatch(LSTMFilter filter, const float *input, float* output);

void LSTMFilterCalculateGradients(LSTMFilter filter, LSTMGradients *gradients, float *d_out);

void LSTMFilterDestroy(LSTMFilter filter);


#if defined __cplusplus
}
#endif


#endif /* Header_h */
