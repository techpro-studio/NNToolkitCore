//
// Created by Alex on 02.11.2020.
//

#include "nntoolkitcore/train/optimizers.h"
#include "nntoolkitcore/core/ops.h"
#include "string.h"
#include "stdlib.h"

void sum_batch_gradient(float * gradients, float* gradient, int size, int batch){
    float* buffer = malloc(size * sizeof(float));
    memset(buffer, 0, size * sizeof(float));
    for (int b = 0; b < batch; ++b) {
        op_vec_add(gradient, gradients + b * size, gradient, size);
    }
    free(buffer);
}

int sgd_optimize(SGD optimizer, float *gradient, float *weights, int size) {
    float* buffer = malloc(size * sizeof(float));
    op_vec_mul_sc(buffer, optimizer.learning_rate, buffer, size);
    op_vec_sub(weights, buffer, weights, size);
    free(buffer);
    return 0;
}

