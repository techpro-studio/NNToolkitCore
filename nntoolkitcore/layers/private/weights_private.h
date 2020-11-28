//
// Created by Alex on 28.11.2020.
//

#ifndef gradient_sum_h
#define gradient_sum_h
#include "nntoolkitcore/layers/shared.h"

DefaultWeights *default_weights_create(DefaultWeightsSize sizes);

void default_weights_destroy(DefaultWeights *weights);

DefaultGradient *default_gradient_create(
    DefaultWeightsSize sizes,
    int input_size
);

void default_gradient_destroy(DefaultGradient *gradient);

void default_gradient_sum(DefaultGradient **gradients, DefaultGradient *gradient, DefaultWeightsSize sizes, int size);



#endif //gradient_sum_h
