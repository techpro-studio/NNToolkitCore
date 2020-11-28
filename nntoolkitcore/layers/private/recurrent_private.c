//
// Created by Alex on 20.11.2020.
//

#include "nntoolkitcore/core/memory.h"
#include "recurrent_private.h"
#include "stdlib.h"
#include "nntoolkitcore/core/ops.h"

RecurrentGradient *recurrent_gradient_create(
    RecurrentWeightsSize sizes,
    int input_size
) {
    RecurrentGradient *gradient = malloc(sizeof(RecurrentGradient));
    int buffer_size = sizes.sum + input_size;
    gradient->d_W = f_malloc(buffer_size);
    gradient->d_U = gradient->d_W + sizes.w;
    gradient->d_b_i = gradient->d_U + sizes.u;
    gradient->d_b_h = gradient->d_b_i + sizes.b_i;
    gradient->d_X = gradient->d_b_h + sizes.b_h;
    return gradient;
}

void recurrent_gradient_destroy(RecurrentGradient *gradient) {
    free(gradient->d_W);
    free(gradient);
}

RecurrentWeights *recurrent_weights_create(RecurrentWeightsSize sizes) {
    RecurrentWeights *weights = malloc(sizeof(RecurrentWeights));
    weights->W = f_malloc(sizes.sum);
    weights->U = weights->W + sizes.w;
    weights->b_i = weights->U + sizes.u;
    weights->b_h = weights->b_i + sizes.b_i;
    return weights;
}

void recurrent_weights_destroy(RecurrentWeights *weights) {
    free(weights->W);
    free(weights);
}

void
recurrent_gradient_sum(RecurrentGradient *current, RecurrentGradient *root, RecurrentWeightsSize sizes, int batch) {
    op_vec_add(root->d_W + batch * sizes.w, current->d_W + batch * sizes.w, root->d_W + batch * sizes.w, sizes.w);
    op_vec_add(root->d_U + batch * sizes.u, current->d_U + batch * sizes.u, root->d_U + batch * sizes.u, sizes.u);
    op_vec_add(root->d_b_i + batch * sizes.b_i, current->d_b_i + batch * sizes.b_i, root->d_b_i + batch * sizes.b_i,
               sizes.b_i);
    op_vec_add(root->d_b_h + batch * sizes.b_h, current->d_b_h + batch * sizes.b_h, root->d_b_h + batch * sizes.b_h,
               sizes.b_h);
}
