//
// Created by Alex on 20.11.2020.
//

#ifndef recurrent_private_h
#define recurrent_private_h

#include "nntoolkitcore/layers/recurrent.h"

RecurrentWeights *recurrent_weights_create(RecurrentWeightsSize sizes);

void recurrent_weights_destroy(RecurrentWeights *weights);

RecurrentGradient *recurrent_gradient_create(
    RecurrentWeightsSize sizes,
    int input_size
);

void recurrent_gradient_destroy(RecurrentGradient *gradient);

void recurrent_gradient_sum(RecurrentGradient *current, RecurrentGradient *root, RecurrentWeightsSize sizes);

#endif //recurrent_private_h
