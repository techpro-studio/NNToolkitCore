//
// Created by Alex on 23.11.2020.
//

#ifndef bidirectional_h
#define bidirectional_h

#if defined __cplusplus
extern "C" {
#endif

#include "recurrent.h"

void bidirectional_reverse_batch(
    const float* input,
    float *output,
    int size,
    int batch
);

//MERGE CONCAT

void bidirectional_merge_concat(
    const float *forward_result,
    const float *backward_result,
    float *output,
    RecurrentConfig config,
    int batch
);

void bidirectional_merge_concat_gradient(
    const float *d_out,
    float *d_forward_out,
    float *d_backward_out,
    RecurrentConfig config,
    int batch
);

//MERGE SUM

void bidirectional_merge_sum(
    const float *forward_result,
    const float *backward_result,
    float *output,
    RecurrentConfig config,
    int batch
);

void bidirectional_merge_sum_gradient(
    const float *d_out,
    float *d_forward_out,
    float *d_backward_out,
    RecurrentConfig config,
    int batch
);


#if defined __cplusplus
}
#endif

#endif //bidirectional_h
