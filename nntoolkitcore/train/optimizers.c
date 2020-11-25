//
// Created by Alex on 02.11.2020.
//

#include "nntoolkitcore/train/optimizers.h"
#include "nntoolkitcore/core/ops.h"
#include "nntoolkitcore/core/memory.h"
#include "stdlib.h"

void sum_batch_gradient(float * gradients, float* gradient, int size, int batch){
    for (int b = 0; b < batch; ++b) {
        op_vec_add(gradient, gradients + b * size, gradient, size);
    }
}

int sgd_optimize(SGD optimizer, float *gradient, float *weights, int size) {
    float* buffer = f_malloc(size);
    op_vec_mul_sc(gradient, optimizer.learning_rate, buffer, size);
    op_vec_sub(weights, buffer, weights, size);
    free(buffer);
    return 0;
}

