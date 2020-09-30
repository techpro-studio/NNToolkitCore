//
//  recurrent_shared.h
//  audio_test
//
//  Created by Alex on 29.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#ifndef recurrent_shared_h
#define recurrent_shared_h


#include <stdio.h>
#include "activation.h"
#include <stdbool.h>

#if defined __cplusplus
extern "C" {
#endif

void ComputeGate(int in, int out, ActivationFunction* activation, const float *x, const float*h, const float *W, const float *U, const float* b_i, const float* b_h, bool useHiddenBias, float* gate);

#if defined __cplusplus
}
#endif

#endif /* recurrent_shared_h */
