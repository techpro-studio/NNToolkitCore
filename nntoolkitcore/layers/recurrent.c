//
// Created by Alex on 28.11.2020.
//
#include "recurrent.h"
#include "nntoolkitcore/layers/private/recurrent_private.h"

RecurrentConfig RecurrentConfigCreate(
        int input_feature_channels,
        int output_feature_channels,
        bool return_sequences,
        int timesteps
) {
    return (RecurrentConfig) {
            .input_feature_channels = input_feature_channels,
            .timesteps = timesteps,
            .return_sequences = return_sequences,
            .output_feature_channels = output_feature_channels
    };
}

void RecurrentGradientDestroy(RecurrentGradient *gradient) {
    recurrent_gradient_destroy(gradient);
}
