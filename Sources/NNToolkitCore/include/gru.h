//
//  gru.h
//  audio_test
//
//  Created by Alex on 21.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef gru_h
#define gru_h

#include <stdio.h>
#include "stdbool.h"
#import "activation.h"


#if defined __cplusplus
extern "C" {
#endif



typedef struct {
    int inputFeatureChannels;

    int outputFeatureChannels;
    /*
     if true:
     h_t = (1 - z) * h_t_previous + z * h_tilda.
     else:
     h_t = (1 - z) * h_tilda + z * h_t_previous
     */
    bool flipOutputGates;

    bool v2;

    bool returnSequences;

    int timesteps;

    ActivationFunction reccurrentActivation;

    ActivationFunction activation;
} GRUConfig;

GRUConfig GRUConfigCreate(int inputFeatureChannels, int outputFeatureChannels, bool flipOutputGates, bool v2, bool returnSequences, int batchSize, ActivationFunction reccurrentActivation, ActivationFunction activation);


typedef struct {
    float *Wz;
    float *Uz;

    float *b_iz;
    float *b_hz;

    float *Wr;
    float *Ur;

    float *b_ir;
    float *b_hr;

    float *Wh;
    float *Uh;
    
    float *b_ih;
    float *b_hh;

} GRUWeights;


typedef struct GRUFilterStruct * GRUFilter;

GRUWeights* GRUFilterGetWeights(GRUFilter filter);

GRUFilter GRUFilterCreate(GRUConfig config);

//feature channels in row

int GRUFilterApply(GRUFilter filter, const float *input, float* output);

void GRUFilterDestroy(GRUFilter filter);


#if defined __cplusplus
}
#endif

#endif /* gru_h */
