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
    config.input_feature_channels = input_feature_channels;
    config.timesteps = timesteps;
    config.return_sequences = return_sequences;
    config.output_feature_channels = output_feature_channels;
    config.activations = activations;
    return config;
}

typedef RecurrentWeightsSize GRUWeightsSize;

typedef struct {

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
    int in = config.input_feature_channels;
    int out = config.output_feature_channels;
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
    filter->h = f_malloc(config.output_feature_channels);
    filter->training_data = NULL;
    filter->inference_data = NULL;
    return filter;
}

void gru_training_data_destroy(GRUTrainingData *data){
    free(data);
}

void gru_inference_data_destroy(GRUInferenceData *data){
    free(data->computation_buffer);
    free(data);
}

GRUInferenceData *gru_inference_data_create(GRUConfig config){
    GRUInferenceData *data = malloc(sizeof(GRUInferenceData));
    data->computation_buffer = f_malloc(13 * config.output_feature_channels);
    return data;
}

GRUTrainingData *gru_training_data_create(GRUConfig config, RecurrentTrainingConfig training_config){
    GRUTrainingData *data = malloc(sizeof(GRUTrainingData));
    int out = config.output_feature_channels;
    int batch = training_config.mini_batch_size;
    int input_size = batch * config.input_feature_channels * config.timesteps;
    int output_size = batch * out * config.timesteps;

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
    float *Z_zr = h_pr_U + 3 * out;
    op_vec_add(x_W, h_pr_U, Z_zr, 2 * out);
    // z = recurrent_activation(Z_zr[0: out])
    // r = recurrent_activation(Z_zr[out: 2 * out])
    float *z = Z_zr + 2 * out;
    float *r = z + out;
    ActivationFunctionApply(activations.input_gate_activation, Z_zr, z);
    ActivationFunctionApply(activations.reset_gate_activation, Z_zr + out, r);
    //x * W_h
    float *h_tilda = r + out;
    //h_pr_U[2* out : 3 * out] <*> r
    op_vec_mul(r, h_pr_U + 2 * out, h_tilda, out);
    //h_tilda += x_W[2 * out: 3 * out]
    op_vec_add(h_tilda, x_W + 2 * out, h_tilda, out);
    //tanh(a_H);
    ActivationFunctionApply(activations.update_gate_activation, h_tilda, h_tilda);
//    filter->config.activation(h_tilda, h_tilda, out);
    // h_t = (1 - z) <*> h_pr + z <*> h_tilda;
    // ht = -z;
    float *minus_z_pw = h_tilda + out;
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
    int out = filter->config.output_feature_channels;
    int in = filter->config.input_feature_channels;
    for (int i = 0; i < filter->config.timesteps; ++i) {
        int output_offset = filter->config.return_sequences ? i * out : 0;
        GRUCellForward(filter->weights, filter->config.activations, in, out, input + i * in, filter->h,
                       output + output_offset, filter->inference_data->computation_buffer);
        f_copy(filter->h, output + output_offset, out);
    }
    return 0;
}

GRUActivations GRUActivationsCreateDefault(int size) {
    GRUActivations result;
    result.input_gate_activation = ActivationFunctionCreateSigmoid(size);
    result.reset_gate_activation = ActivationFunctionCreateSigmoid(size);
    result.update_gate_activation = ActivationFunctionCreateTanh(size);
    return result;
}

void GRUActivationsDestroy(GRUActivations activations) {
    ActivationFunctionDestroy(activations.input_gate_activation);
    ActivationFunctionDestroy(activations.reset_gate_activation);
    ActivationFunctionDestroy(activations.update_gate_activation);
}

GRUActivations GRUActivationsCreate(ActivationFunction input_gate_activation, ActivationFunction update_gate_activation,
                                    ActivationFunction reset_gate_activation) {

    GRUActivations activations;
    activations.update_gate_activation = update_gate_activation;
    activations.input_gate_activation = input_gate_activation;
    activations.reset_gate_activation = reset_gate_activation;
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
        training_config.mini_batch_size,
        config.input_feature_channels * config.timesteps
    );
}

int GRUApplyTrainingBatch(GRU filter, const float *input, float *output) {
    return 0;
}

void GRUCalculateGradient(GRU filter, GRUGradient *gradients, float *d_out) {

}


