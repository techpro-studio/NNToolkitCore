//
// Created by Alex on 23.11.2020.
//

#ifndef bidirectional_h
#define bidirectional_h

#if defined __cplusplus
extern "C" {
#endif

#include "recurrent.h"

void bd_reverse_input_batch(
    const float* input,
    float *output,
    RecurrentConfig config,
    int batch
);

//MERGE CONCAT

int bd_merge_concat_buffer_size(RecurrentConfig config);

void bd_merge_concat(
    const float *forward_result,
    const float *backward_result,
    float *output,
    RecurrentConfig config,
    int batch,
    float *buffer
);

void bd_merge_concat_gradient(
    const float *d_out,
    float *d_forward_out,
    float *d_backward_out,
    RecurrentConfig config,
    int batch,
    float *buffer
);

//MERGE SUM

void bd_merge_sum(
    const float *forward_result,
    const float *backward_result,
    float *output,
    RecurrentConfig config,
    int batch
);

void bd_merge_sum_gradient(
    const float *d_out,
    float *d_forward_out,
    float *d_backward_out,
    RecurrentConfig config,
    int batch
);

void bd_accumulate_d_x(
    const float* forward_dx,
    const float* backward_dx,
    float *output,
    RecurrentConfig config,
    int batch
);

#if defined __cplusplus
}
#endif

#endif //bidirectional_h
