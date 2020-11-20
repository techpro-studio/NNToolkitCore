//
// Created by Alex on 20.11.2020.
//

#include "nntoolkitcore/core/memory.h"
#include "recurrent_private.h"
#include "stdlib.h"

RecurrentGradient *recurrent_gradient_create(
        RecurrentWeightsSize sizes,
        RecurrentTrainingConfig training_config,
        int input_size
) {
    RecurrentGradient *gradient = malloc(sizeof(RecurrentGradient));
    int batch = training_config.mini_batch_size;
    int buffer_size = sizes.sum * batch + batch * input_size;
    gradient->d_W = f_malloc(buffer_size);
    gradient->d_U = gradient->d_W + sizes.w * batch;
    gradient->d_b_i = gradient->d_U + sizes.u * batch;
    gradient->d_b_h = gradient->d_b_i + sizes.b_i * batch;
    gradient->d_X = gradient->d_b_h + sizes.b_h * batch;
    return gradient;
}

void recurrent_gradient_destroy(RecurrentGradient *gradient) {
    free(gradient->d_W);
    free(gradient);
}

RecurrentWeights *recurrent_weights_create(RecurrentWeightsSize sizes) {
    float *buffer = f_malloc(sizes.sum);
    return NULL;
}

void recurrent_weights_destroy(RecurrentWeights *weights) {

}
