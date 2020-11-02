//
//  File.c
//  
//
//  Created by Alex on 17.10.2020.
//

#include "loss.h"
#include "operations.h"


float mean_squared_error(float* y, float * y_pred, int size, int batch) {
    float loss = 0.0f;
    S_LOOP_START(batch, b)
        float buffer[size];
        float cost = 0.0f;
        op_vec_sub(y + b * size, y_pred + b * size, buffer + b * size, size);
        op_vec_mul(buffer + b * size, buffer + b * size, buffer + b * size, size);
        op_vec_sum(buffer + b * size, &cost, size);
        loss += cost / size;
    S_LOOP_END
    return loss / batch;
}

void mean_squared_error_derivative(float* y, float * y_pred, float *d_y_pred, int size, int batch) {
    P_LOOP_START(batch, b)
        op_vec_sub(y + b * size, y_pred + b * size, d_y_pred + b * size, size);
        float k = -2.0 / size;
        op_vec_mul_sc(d_y_pred + b * size, k, d_y_pred + b * size, size);
    P_LOOP_END
}
