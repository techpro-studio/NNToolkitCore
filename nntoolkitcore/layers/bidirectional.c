//
// Created by Alex on 23.11.2020.
//

#include "bidirectional.h"
#include "nntoolkitcore/core/loop.h"
#include "nntoolkitcore/core/ops.h"
#include "nntoolkitcore/core/memory.h"

void reverse(const float *input, float *output, int size){
    for (int f = 0; f < size; ++f){
        output[f] = input[size - f - 1];
    }
}

void bd_reverse_batch(const float *input, float *output, int size, int batch) {
    P_LOOP_START(batch, b)
        reverse(input + b * size, output + b * size, size);
    P_LOOP_END
}

void bd_merge_concat(
    const float *forward_result,
    const float *backward_result,
    float *output,
    RecurrentConfig config,
    int batch
) {
    int rows = config.return_sequences ? config.timesteps : 1;
    int columns = config.output_feature_channels;
    int size = rows * columns;
    for (int b = 0; b < batch; ++b) {
        float buffer[2 * rows * columns];
        op_mat_transp(forward_result + b * size, buffer, columns, rows);
        op_mat_transp(backward_result + b * size, buffer + size, columns, rows);
        op_mat_transp(buffer, output + 2 * b * size, rows, 2 * columns);
    }
}

void bd_merge_concat_gradient(
    const float *d_out,
    float *d_forward_out,
    float *d_backward_out,
    RecurrentConfig config,
    int batch
) {
    int rows = config.return_sequences ? config.timesteps : 1;
    int columns = config.output_feature_channels;
    int size = rows * columns;
    float buffer[2 * rows * columns];
    for (int b = 0; b < batch; ++b) {
        op_mat_transp(d_out + 2 * b * size, buffer, 2 * columns, rows);
        op_mat_transp(buffer, d_forward_out + b * size, rows, columns);
        op_mat_transp(buffer + size, d_backward_out + b * size, rows, columns);
    }
}

void bd_merge_sum(
    const float *forward_result,
    const float *backward_result,
    float *output,
    RecurrentConfig config,
    int batch
) {
    int rows = config.return_sequences ? config.timesteps : 1;
    op_vec_add(forward_result, backward_result, output, batch * config.output_feature_channels * rows);
}

void bd_merge_sum_gradient(
    const float *d_out,
    float *d_forward_out,
    float *d_backward_out,
    RecurrentConfig config,
    int batch
) {
    int rows = config.return_sequences ? config.timesteps : 1;
    f_copy(d_forward_out, d_out, batch * rows * config.output_feature_channels);
    f_copy(d_backward_out, d_out, batch * rows * config.output_feature_channels);
}

void bd_accumulate_d_x(
    const float *forward_dx,
    const float *backward_dx,
    float *output,
    int size,
    int batch
) {
    bd_reverse_batch(backward_dx, output, size, batch);
    op_vec_add(forward_dx, output, output, size * batch);
}
