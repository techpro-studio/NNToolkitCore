//
// Created by Alex on 05.11.2020.
//

#include <nntoolkitcore/layers/private/recurrent_private.h>
#include "nntoolkitcore/layers/gru.h"
#include "nntoolkitcore/core/ops.h"
#include "nntoolkitcore/core/memory.h"
#include "stdlib.h"
#include "activation_default.h"

GRUConfig
GRUConfigCreate(int input_feature_channels, int output_feature_channels, bool return_sequences, int timesteps,
                GRUActivations activations) {
    GRUConfig config;
    config.base = RecurrentConfigCreate(input_feature_channels, output_feature_channels, return_sequences, timesteps);
    config.activations = activations;
    return config;
}

typedef RecurrentWeightsSize GRUWeightsSize;

typedef struct {
    GRUTrainingConfig config;
    float *input;
    float *output;
    float *d_H;
    float *h_pr_Uh;
    float* Z_gates;
    float *computation_buffer;
    RecurrentGradient *current_weights_gradient;
} GRUTrainingData;

typedef struct {
    float *computation_buffer;
}GRUInferenceData;


struct GRUStruct {
    GRUConfig config;
    float *h;
    GRUWeights *weights;
    GRUTrainingData *training_data;
    GRUInferenceData *inference_data;
};

GRUWeights *GRUGetWeights(GRU filter) {
    return filter->weights;
}

GRUWeightsSize gru_weights_size_from_config(GRUConfig config){
    int in = config.base.input_feature_channels;
    int out = config.base.output_feature_channels;
    GRUWeightsSize size;
    size.w = in * out * 3;
    size.u = out * out * 3;
    size.b_i = out * 3;
    size.b_h = out * 3;
    size.sum = size.w + size.u + size.b_h + size.b_i;
    return size;
}

GRU gru_create(GRUConfig config){
    GRU filter = malloc(sizeof(struct GRUStruct));
    filter->config = config;
    filter->weights = recurrent_weights_create(gru_weights_size_from_config(config));
    filter->h = f_malloc(config.base.output_feature_channels);
    filter->training_data = NULL;
    filter->inference_data = NULL;
    return filter;
}

void gru_training_data_destroy(GRUTrainingData *data){
    free(data->input);
    recurrent_gradient_destroy(data->current_weights_gradient);
    free(data);
}

void gru_inference_data_destroy(GRUInferenceData *data){
    free(data->computation_buffer);
    free(data);
}

GRUInferenceData *gru_inference_data_create(GRUConfig config){
    GRUInferenceData *data = malloc(sizeof(GRUInferenceData));
    data->computation_buffer = f_malloc(14 * config.base.output_feature_channels);
    return data;
}

GRUTrainingData *gru_training_data_create(GRUConfig config, RecurrentTrainingConfig training_config){
    GRUTrainingData *data = malloc(sizeof(GRUTrainingData));
    int out = config.base.output_feature_channels;
    int batch = training_config.mini_batch_size;
    int input_size = batch * config.base.input_feature_channels * config.base.timesteps;
    int output_size = batch * out * config.base.timesteps;
    // input + output+ Z_gates + d_h + buffer
    int training_buffer = input_size + 8 * output_size + batch * out + 8 * out;
    //input size 0 since we need w8s only
    data->current_weights_gradient = recurrent_gradient_create(gru_weights_size_from_config(config), 0);
    data->config = training_config;
    data->input = f_malloc(training_buffer);
    data->output = data->input + input_size;
    data->Z_gates = data->output + output_size;
    data->h_pr_Uh = data->Z_gates + 6 * output_size;
    data->d_H = data->h_pr_Uh + output_size;
    data->computation_buffer = data->d_H + batch * out;
    return data;
}

GRU GRUCreateForInference(GRUConfig config) {
    GRU filter = gru_create(config);
    filter->inference_data = gru_inference_data_create(config);
    return filter;
}

void GRUDestroy(GRU filter) {
    recurrent_weights_destroy(filter->weights);
    if (filter->training_data != NULL){
        gru_training_data_destroy(filter->training_data);
    }
    if (filter->inference_data != NULL){
        gru_inference_data_destroy(filter->inference_data);
    }
    free(filter->h);
    free(filter);
}


static void GRUCellForward(
    GRUWeights *weights,
    GRUActivations activations,
    int in,
    int out,
    const float *x,
    const float *h_pr,
    float *ht,
    float* Z_gates,
    float *buffer
) {
    // W = [Wz, Wr, Wh]
    // U = [Uz, Ur, Uh]
    // x_W = x * W
    float *x_W = buffer;
    op_mat_mul(x, weights->W, x_W, 1, 3 * out, in);
    // b = [bz, br, bh]
    // x_W += bi
    op_vec_add(x_W, weights->b_i, x_W, 3 * out);

    float *h_pr_U = x_W + 3 * out;
    op_mat_mul(h_pr, weights->U, h_pr_U, 1, 3 * out, out);
    op_vec_add(h_pr_U, weights->b_h, h_pr_U, 3 * out);
    // Z_zr = x_W[0: 2 * out] + h_pr_U[0: 2 * out]

    // 6 * out; (Z_z, Z_r, Z_h_tilda, z, r, h_tilda)
    float *Z_zr = Z_gates;
    op_vec_add(x_W, h_pr_U, Z_zr, 2 * out);
    // z = recurrent_activation(Z_zr[0: out])
    // r = recurrent_activation(Z_zr[out: 2 * out])

    float *z = Z_zr + 3 * out;
    float *r = z + out;
    ActivationFunctionApply(activations.z_gate_activation, Z_zr, z);
    ActivationFunctionApply(activations.r_gate_activation, Z_zr + out, r);

    //x * W_h
    float * Z_h_tilda = Z_zr + 2 * out;
    float *h_tilda = r + out;
    //h_pr_U[2* out : 3 * out] <*> r
    op_vec_mul(r, h_pr_U + 2 * out, Z_h_tilda, out);
    //h_tilda += x_W[2 * out: 3 * out]
    op_vec_add(Z_h_tilda, x_W + 2 * out, Z_h_tilda, out);
    //tanh(a_H);
    ActivationFunctionApply(activations.h_gate_activation, Z_h_tilda, h_tilda);
//    filter->config.activation(h_tilda, h_tilda, out);
    // h_t = (1 - z) <*> h_pr + z <*> h_tilda;
    // ht = -z;
    float *minus_z_pw = h_pr_U + 3 * out;
    op_vec_neg(z, minus_z_pw, out);
    //ht= -z + 1
    op_vec_add_sc(minus_z_pw, 1, minus_z_pw, out);
    //ht = (1 - z) <*> h_tilda
    op_vec_mul(minus_z_pw, h_tilda, minus_z_pw, out);
    //h_tilda = z <*> h_pr
    float *z_h_pw = minus_z_pw + out;
    op_vec_mul(z, h_pr, z_h_pw, out);
    op_vec_add(minus_z_pw, z_h_pw, ht, out);
}

int GRUApplyInference(GRU filter, const float *input, float *output) {
    if(filter->training_data != NULL){
        return -1;
    }
    int out = filter->config.base.output_feature_channels;
    int in = filter->config.base.input_feature_channels;
    for (int i = 0; i < filter->config.base.timesteps; ++i) {
        int output_offset = filter->config.base.return_sequences ? i * out : 0;
        GRUCellForward(filter->weights, filter->config.activations, in, out, input + i * in, filter->h,
                       output + output_offset,
                       filter->inference_data->computation_buffer,
                       filter->inference_data->computation_buffer + 6 * out);
        f_copy(filter->h, output + output_offset, out);
    }
    return 0;
}

GRUActivations GRUActivationsCreateDefault(int size) {
    GRUActivations result;
    result.z_gate_activation = ActivationFunctionCreateSigmoid(size);
    result.r_gate_activation = ActivationFunctionCreateSigmoid(size);
    result.h_gate_activation = ActivationFunctionCreateTanh(size);
    return result;
}

void GRUActivationsDestroy(GRUActivations activations) {
    ActivationFunctionDestroy(activations.z_gate_activation);
    ActivationFunctionDestroy(activations.r_gate_activation);
    ActivationFunctionDestroy(activations.h_gate_activation);
}

GRUActivations GRUActivationsCreate(
    ActivationFunction z_gate_activation,
    ActivationFunction h_gate_activation,
    ActivationFunction r_gate_activation
) {
    GRUActivations activations;
    activations.h_gate_activation = h_gate_activation;
    activations.z_gate_activation = z_gate_activation;
    activations.r_gate_activation = r_gate_activation;
    return activations;
}

GRU GRUCreateForTraining(GRUConfig config, GRUTrainingConfig training_config) {
    GRU filter = gru_create(config);
    filter->training_data = gru_training_data_create(config, training_config);
    return filter;
}

GRUGradient *GRUGradientCreate(GRUConfig config, GRUTrainingConfig training_config) {
    return recurrent_gradient_create(
        gru_weights_size_from_config(config),
        training_config.mini_batch_size *
        config.base.input_feature_channels * config.base.timesteps
    );
}

int GRUApplyTrainingBatch(GRU filter, const float *input, float *output) {
    if (filter->training_data == NULL){
        return -1;
    }

    int out = filter->config.base.output_feature_channels;
    int in = filter->config.base.input_feature_channels;
    int batch = filter->training_data->config.mini_batch_size;
    int ts = filter->config.base.timesteps;

    int input_buffer_size = batch * ts * in;
    f_copy(filter->training_data->input, input, input_buffer_size);

    for (int b = 0; b < batch; ++b){
        f_zero(filter->h, out);
        for (int i = 0; i < filter->config.base.timesteps; ++i){
            int t_out_offset = out * i + b * ts * out;
            const float *x_t = input + i * in + b * ts * in;
            float *h_t = filter->training_data->output + t_out_offset;
            float *Z_gates = filter->training_data->Z_gates + 6 * t_out_offset;
            float *h_t_prev = filter->h;
            float *h_pr_Uh = filter->training_data->h_pr_Uh + t_out_offset;

            GRUCellForward(
                filter->weights,
                filter->config.activations,
                in, out,
                x_t,
                h_t_prev,
                h_t,
                Z_gates,
                filter->training_data->computation_buffer
            );
            f_copy(h_pr_Uh, filter->training_data->computation_buffer + 5 * out, out);
            f_copy(filter->h, h_t, out);
            f_zero(filter->training_data->computation_buffer, 8 * out);
        }
    }
    if (filter->config.base.return_sequences){
        f_copy(output, filter->training_data->output, batch * ts * out);
    } else {
        for (int b = 0; b < batch; ++b){
            int offset = ((ts - 1) * out) + b * ts * out;
            f_copy(output + b * out, filter->training_data->output + offset, out);
        }
    }
    return 0;
}

typedef struct {
    float *h_t;
    float *x_t;
    float *h_pr_Uh;
    float *h_t_prev;
    float *Z_gates;
} CellBackwardCache;

typedef struct {
    float *d_h_t_prev;
    float *d_x_t;
    float *d_W_t;
    float *d_U_t;
    float *d_bi_t;
    float *d_bh_t;
} CellBackwardGradients;

void
GRUCellBackward(GRUWeights *weights, GRUActivations activations, int in, int out, float *d_h_t, CellBackwardCache cache,
                CellBackwardGradients gradients, float *buffer) {

    // Convenient preparation;
    float* Z_z_gate = cache.Z_gates;
    float* Z_r_gate = cache.Z_gates + out;
    float* Z_h_tilda_gate = cache.Z_gates + 2 * out;
    float* z_gate = cache.Z_gates + 3 * out;
    float* r_gate = cache.Z_gates + 4 * out;
    float* h_tilda_gate = cache.Z_gates + 5 * out;

    float* d_x_W = buffer; // 3 out;
    float* d_h_pr_U = d_x_W + 3 * out;
    float* d_b_i = d_h_pr_U + 3 * out;
    float* d_b_h = d_b_i + 3 * out;

    /*
     * 1. Forward step
     *    h_t = (1 - z_t) * h_tilda + z_t * h_prev;
     *    Backward <- d_h_t;
     *    d_h_prev_1 = z_t * d_h_t;
     *    d_h_tilda = (1 - z_t) * d_h_t;
     *    d_z_t = (h_prev - h_tilda) * d_h_t
     * */

    float* d_h_prev_1 = d_b_h + 3 * out;
    op_vec_mul(z_gate, d_h_t, d_h_prev_1, out);

    float* d_h_tilda = d_h_prev_1 + out;
    op_vec_neg(z_gate, d_h_tilda, out);
    op_vec_mul(d_h_tilda, d_h_t, d_h_tilda, out);
    op_vec_add(d_h_tilda, d_h_t, d_h_tilda, out);

    float* d_z_t = d_h_tilda + out;
    if (cache.h_t_prev){
        op_vec_sub(cache.h_t_prev, h_tilda_gate, d_z_t, out);
    } else{
        op_vec_neg(h_tilda_gate, d_z_t, out);
    }
    op_vec_mul(d_z_t, d_h_t, d_z_t, out);

    /*
     * 2. Forward step
     *    h_tilda = cand_act(x*Wh + b_h + r_t(h_t_prev * U_h + b_hh))
     *
     *    Backward <- d_h_tilda
     *    d_z_h_tilda = d_cand_act * d_h_tilda;
     *    d(x * Wh) = d_z_h_tilda;
     *    d_b_ih = d_z_h_tilda;
     *    d_r_t = (h_t_prev * U_h + b_h_h) * d_z_h_tilda;
     *    d_b_h_h = d(h_t_prev * U_h) = r_t * d_z_h_tilda;
     * */
    float* d_z_h_tilda = d_z_t + out;
    ActivationFunctionCalculateGradient(
        activations.h_gate_activation,
        Z_h_tilda_gate, h_tilda_gate,
        d_h_tilda, d_z_h_tilda
    );

    float* d_r_t = d_z_h_tilda + out;
    op_vec_mul(cache.h_pr_Uh, d_z_h_tilda, d_r_t, out);

    //d(x * Wh)
    f_copy(d_x_W + 2 * out, d_z_h_tilda, out);
    //d_b_i_h
    f_copy(d_b_i + 2 * out, d_z_h_tilda, out);
    //d_h_pr_UH
    op_vec_mul(r_gate, d_z_h_tilda, d_h_pr_U + 2 * out, out);
    //d_b_h_h = d_h_pr_UH
    f_copy(d_b_h + 2 * out, d_h_pr_U + 2 * out, out);
    /*
     * 3. Forward step
     *   r_t = r_act(x * Wr + b_r + h_pr * Ur + b_hr)
     *   Backward <- d_r_t
     *
     *   d_z_r_t = d_r_t * d_r_act;
     *
     *   d(x*Wr) = d_b_i_r = d(h_pr * Ur) = d_b_hr = d_z_r_t;
     *
     *    4. Forward step
     *   z_t = z_act(x * Wz + b_z + h_pr * Uz + b_hz)
     *   Backward <- d_z_t
     *
     *   d_z_z_t = d_z_t * d_z_act;
     *
     *   d(x*Wz) = d_b_i_z = d(h_pr * Uz) = d_b_hz = d_z_z_t;
     * */
    float* d_Z_zr_t = d_r_t + out;
    float *d_z_z_t = d_Z_zr_t;
    float *d_z_r_t = d_Z_zr_t + out;

    ActivationFunctionCalculateGradient(
        activations.z_gate_activation,
        Z_z_gate, z_gate,
        d_z_t, d_z_z_t
    );
    ActivationFunctionCalculateGradient(
        activations.r_gate_activation,
        Z_r_gate, r_gate,
        d_r_t, d_z_r_t
    );

    f_copy(d_b_i, d_Z_zr_t, 2 * out);
    f_copy(d_b_h, d_Z_zr_t, 2 * out);
    f_copy(d_x_W, d_Z_zr_t, 2 * out);
    f_copy(d_h_pr_U, d_Z_zr_t, 2 * out);

    /* 5.
     * first buffer:
     * d(x * W) = d(x * Wz) d(x * Wr) d(x * Wh)
     * second buffer:
     * d(h_pr * U) = d(h_pr * Wz) d*(h_pr * Wr) d(h_pr * Wh)
     *
     * d_bi = d_b_i_z d_b_i_r d_b_i_h;
     * d_bh = d_b_h_z d_b_h_r d_b_h_h;
     * dX = W * d(x * W);
     * dW = x * d(x * W);
     * d_h_pr_2 = U * d(h_pr * U);
     * d_U = h_pr * d(d_pr * U);
     * d_h_pr = d_h_pr_1 + d_h_pr_2;
     * */
    op_mat_mul(weights->W, d_x_W, gradients.d_x_t, in, 1, 3 * out);
    float* d_h_prev_2 = d_Z_zr_t + 2 * out;
    op_mat_mul(weights->U, d_h_pr_U, d_h_prev_2, out, 1, 3 * out);
    op_vec_add(d_h_prev_1, d_h_prev_2, gradients.d_h_t_prev, out);
    op_mat_mul(cache.x_t, d_x_W, gradients.d_W_t, in, 3 * out, 1);
    f_copy(gradients.d_bi_t, d_b_i, 3 * out);
    f_copy(gradients.d_bh_t, d_b_h, 3 * out);
    if (cache.h_t_prev){
        op_mat_mul(cache.h_t_prev, d_h_pr_U, gradients.d_U_t, out, 3 * out, 1);
    } else {
        f_zero(gradients.d_U_t, 3 * out * out);
    }
}

void GRUCalculateGradient(GRU filter, GRUGradient *gradient, float *d_out) {
    if (filter->training_data == NULL){
        return;
    }

    int batch = filter->training_data->config.mini_batch_size;
    int ts = filter->config.base.timesteps;
    int in = filter->config.base.input_feature_channels;
    int out = filter->config.base.output_feature_channels;

    float *h = filter->training_data->output;
    float *x = filter->training_data->input;
    float *Z_gates = filter->training_data->Z_gates;
    float *h_pr_Uh = filter->training_data->h_pr_Uh;

    GRUWeightsSize sizes = gru_weights_size_from_config(filter->config);
    float *dh = filter->training_data->d_H;

    for (int b = 0; b < batch; ++b) {
        for (int t = ts - 1; t >= 0; --t) {
            size_t t_out_offset = t * out + b * ts * out;

            CellBackwardCache cache;

            cache.h_t_prev = t == 0 ? NULL : h + (t_out_offset - out);
            cache.x_t = x + t * in + b * ts * in;
            cache.h_t = h + t_out_offset;
            cache.h_pr_Uh = h_pr_Uh + t_out_offset;
            cache.Z_gates = Z_gates + 6 * t_out_offset;

            CellBackwardGradients current_gradients;

            current_gradients.d_W_t = filter->training_data->current_weights_gradient->d_W;
            current_gradients.d_U_t = filter->training_data->current_weights_gradient->d_U;
            current_gradients.d_bi_t = filter->training_data->current_weights_gradient->d_b_i;
            current_gradients.d_bh_t = filter->training_data->current_weights_gradient->d_b_h;

            current_gradients.d_h_t_prev = dh + (b * out);
            current_gradients.d_x_t = gradient->d_X + (t * in + b * ts * in);


            int computation_buffer_size = 20 * out;
            float* computation_buffer = f_malloc(computation_buffer_size);

            float *d_h_t_init = t == ts - 1 ? NULL : dh + (b * out);
            bool seq = filter->config.base.return_sequences;
            float d_out_t[out];
            f_zero(d_out_t, out);
            if (seq) {
                f_copy(d_out_t, d_out + t_out_offset, out);
            } else if (t == ts - 1) {
                f_copy(d_out_t, d_out + b * out, out);
            }
            float d_h_t[out];
            f_zero(d_h_t, out);
            op_vec_add(d_h_t_init == NULL ? d_h_t : d_h_t_init, d_out_t, d_h_t, out);

            GRUCellBackward(filter->weights, filter->config.activations, in, out, d_h_t, cache,
                            current_gradients, computation_buffer);
            f_zero(computation_buffer, computation_buffer_size);

            recurrent_gradient_sum(filter->training_data->current_weights_gradient, gradient, sizes);
        }
    }
}



