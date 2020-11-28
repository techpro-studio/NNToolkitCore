//
// Created by Alex on 28.11.2020.
//

#include "gradient_sum.h"
#include "nntoolkitcore/core/ops.h"

void sum_batch_gradient(float * gradients, float* gradient, int size, int batch){
    for (int b = 0; b < batch; ++b) {
        op_vec_add(gradient, gradients + b * size, gradient, size);
    }
}