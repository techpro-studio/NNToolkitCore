//
// Created by Alex on 23.11.2020.
//

#ifndef bidirectional_h
#define bidirectional_h

#if defined __cplusplus
extern "C" {
#endif

#include "recurrent.h"

void bd_reverse_batch(
    const float* input,
    float *output,
    int size,
    int batch
);

//MERGE CONCAT

void bd_merge_concat(
    const float *forward_result,
    const float *backward_result,
    float *output,
    RecurrentConfig config,
    int batch
);

void bd_merge_concat_gradient(
    const float *d_out,
    float *d_forward_out,
    float *d_backward_out,
    RecurrentConfig config,
    int batch
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
    int size,
    int batch
);

#if defined __cplusplus
}
#endif

#endif //bidirectional_h
