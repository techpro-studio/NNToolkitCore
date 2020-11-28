//
// Created by Alex on 28.11.2020.
//

#include "weights_private.h"
#include "nntoolkitcore/core/ops.h"
#include "stdlib.h"
#include "nntoolkitcore/core/memory.h"

void sum_batch_gradient(float * gradients, float* gradient, int size, int batch){
    for (int b = 0; b < batch; ++b) {
        op_vec_add(gradient, gradients + b * size, gradient, size);
    }
}

DefaultWeights *default_weights_create(DefaultWeightsSize sizes) {
    DefaultWeights *weights = malloc(sizeof(DefaultWeights));
    weights->W = f_malloc(sizes.sum);
    weights->b = weights->W + sizes.w;
    return weights;
}

void default_weights_destroy(DefaultWeights *weights) {
    free(weights->W);
    free(weights);
}


DefaultGradient *default_gradient_create(DefaultWeightsSize sizes, int input_size) {
    DefaultGradient *gradient = malloc(sizeof(DefaultGradient));
    int size = input_size + sizes.sum;
    gradient->d_W = f_malloc(size);
    gradient->d_b = gradient->d_W + sizes.w;
    gradient->d_X = gradient->d_b + sizes.b;
    return gradient;
}

void default_gradient_destroy(DefaultGradient *gradient) {
    free(gradient->d_W);
    free(gradient);
}

void default_gradient_sum(DefaultGradient **gradients, DefaultGradient *gradient, DefaultWeightsSize sizes, int size) {
    for (int i = 0; i < size; ++i){
        op_vec_add(gradients[i]->d_W, gradient->d_W, gradient->d_W, sizes.w);
        op_vec_add(gradients[i]->d_b, gradient->d_b, gradient->d_b, sizes.b);
    }
}

