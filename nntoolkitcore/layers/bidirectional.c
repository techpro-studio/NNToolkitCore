//
// Created by Alex on 23.11.2020.
//

#include "bidirectional.h"
#include "nntoolkitcore/core/loop.h"

void reverse(const float *input, float *output, int size){
    for (int f = 0; f < size; ++f){
        output[f] = input[size - f - 1];
    }
}

void bidirectional_reverse_batch(const float *input, float *output, int size, int batch) {
    P_LOOP_START(batch, b)
        reverse(input + b * size, output + b * size, size);
    P_LOOP_END
}

void bidirectional_merge_concat(
    const float *forward_result,
    const float *backward_result,
    float *output,
    RecurrentConfig config,
    int batch
) {


}

void bidirectional_merge_concat_gradient(
    const float *d_out,
    float *d_forward_out,
    float *d_backward_out,
    RecurrentConfig config,
    int batch
) {

}

void bidirectional_merge_sum(
    const float *forward_result,
    const float *backward_result,
    float *output,
    RecurrentConfig config,
    int batch
) {

}

void bidirectional_merge_sum_gradient(
    const float *d_out,
    float *d_forward_out,
    float *d_backward_out,
    RecurrentConfig config,
    int batch
) {

}
