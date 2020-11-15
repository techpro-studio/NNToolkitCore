//
//  File.c
//  
//
//  Created by Alex on 17.10.2020.
//

#include "nntoolkitcore/train/loss.h"
#include "nntoolkitcore/core/ops.h"
#include "nntoolkitcore/core/loop.h"

float mean_squared_error(float* y, float * y_pred, int size, int batch) {
    float loss = 0.0f;
    for (int b = 0; b < batch; ++b) {
        float buffer[size];
        float one = 0.0f;
        op_vec_sub(y + b * size, y_pred + b * size, buffer, size);
        op_vec_mul(buffer, buffer, buffer, size);
        op_vec_sum(buffer, &one, size);
        loss += one / (float)size;
    }
    return loss / (float)batch;
}

void mean_squared_error_derivative(float* y, float * y_pred, float *d_y_pred, int size, int batch) {
    P_LOOP_START(batch, b)
        op_vec_sub(y + b * size, y_pred + b * size, d_y_pred + b * size, size);
        float k = -2.0f / (float )(size * batch);
        op_vec_mul_sc(d_y_pred + b * size, k, d_y_pred + b * size, size);
    P_LOOP_END
}

float categorical_crossentropy(float *y, float *y_pred, int c, int batch) {
    float loss = 0.0f;
    for (int b = 0; b < batch; ++b) {
        float buffer[c];
        float one = 0.0f;
        op_vec_log(y_pred + b * c, buffer, c);
        op_vec_mul(buffer, y + b * c, buffer, c);
        op_vec_sum(buffer, &one, c);
        loss += -one;
    }
    return loss / (float)batch;
}

void categorical_crossentropy_derivative(float *y, float *y_pred, float *d_y_pred, int c, int batch) {
    P_LOOP_START(batch, b)
        op_vec_div(y, y_pred, d_y_pred, c);
        op_vec_mul_sc(d_y_pred, -1.0f, d_y_pred, c);
    P_LOOP_END
}

