//
//  recurrent_shared.c
//  audio_test
//
//  Created by Alex on 29.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "recurrent_shared.h"
#include "operations.h"


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
    ActivationFunctionApply(activation, gate, gate);
}
