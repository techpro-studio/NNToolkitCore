//
// Created by Alex on 20.11.2020.
//

#include <nntoolkitcore/core/memory.h>
#include "recurrent_base.h"

RecurrentTrainingConfig RecurrentTrainingConfigCreate(int mini_batch_size) {
    RecurrentTrainingConfig result;
    result.mini_batch_size = mini_batch_size;
    return result;
}

RecurrentGradient *RecurrentGradientCreate(
        RecurrentWeightsSize sizes,
        RecurrentTrainingConfig training_config,
        int input_size
) {
    RecurrentGradient *gradient = malloc(sizeof(RecurrentGradient));
    int batch = training_config.mini_batch_size;
    int buffer_size = (sizes.buffer *  batch + batch * input_size) * sizeof(float);
    gradient->d_W = malloc_zeros(buffer_size);
    gradient->d_U = gradient->d_W + sizes.w * batch;
    gradient->d_b_i = gradient->d_U + sizes.u * batch;
    gradient->d_b_h = gradient->d_b_i + sizes.b_i * batch;
    gradient->d_X = gradient->d_b_h + sizes.b_h * batch;
    return gradient;
}

void RecurrentGradientDestroy(RecurrentGradient *gradient) {
    free(gradient->d_W);
    free(gradient);
}
