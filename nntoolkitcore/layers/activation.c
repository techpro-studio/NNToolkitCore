//
//  activation.c
//  audio_test
//
//  Created by Alex on 22.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "nntoolkitcore/layers/activation.h"

#include "stdlib.h"
#include <string.h>

struct ActivationFunctionStruct {
    void *implementer;
    int input_size;
    ActivationImplementerDestroy destroy_fn;
    ActivationFunctionDerivative derivative;
    ActivationFunctionDerivative cached_derivative;
    ActivationFunctionImpl function;
};

void ActivationFunctionApply(ActivationFunction filter, const float *input, float *output) {
    filter->function(filter->implementer, input, output, filter->input_size);
}

void ActivationFunctionDestroy(ActivationFunction filter) {
    if (filter->implementer) {
        filter->destroy_fn(filter->implementer);
    }
    free(filter);
}

ActivationFunction ActivationFunctionCreate(int size, ActivationImplementerDestroy destroy_fn, void *implementer,
                                            ActivationFunctionImpl function, ActivationFunctionDerivative derivative,
                                            ActivationFunctionDerivative cached_derivative) {
    ActivationFunction filter = malloc(sizeof(struct ActivationFunctionStruct));
    filter->implementer = implementer;
    filter->input_size = size;
    filter->destroy_fn = destroy_fn;
    filter->function = function;
    filter->cached_derivative = cached_derivative;
    filter->derivative = derivative;
    return filter;
}

void ActivationFunctionApplyDerivative(ActivationFunction filter, const float *z, const float *a, const float *d_out,
                                       float *output) {
    if (filter->cached_derivative == NULL || a == NULL) {
        filter->derivative(filter->implementer, z, d_out, output, filter->input_size);
    } else {
        filter->cached_derivative(filter->implementer, a, d_out, output, filter->input_size);
    }
}

