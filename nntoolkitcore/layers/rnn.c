//
// Created by Alex on 19.11.2020.
//

#include "nntoolkitcore/core/memory.h"
#include "nntoolkitcore/layers/private/recurrent_private.h"
#include "rnn.h"
#include "stdlib.h"
#include "nntoolkitcore/core/ops.h"


typedef struct {
    RNNTrainingConfig config;
    float *computation_buffer;
    float *gate;
    float *input;
    float *dH;
    float *output;
    RecurrentGradient *current_weights_gradient;
} RNNTrainingData;

typedef struct {
    float *computation_buffer;
} RNNInferenceData;

typedef RecurrentWeightsSize RNNWeightsSize;

struct RNNStruct {
    RNNWeights *weights;
    RNNConfig config;
    float *h;
    RNNInferenceData *inference_data;
    RNNTrainingData *training_data;
};


RNNWeightsSize rnn_weights_size_from_config(RNNConfig config) {
    int in = config.base.input_feature_channels;
    int out = config.base.output_feature_channels;
    RNNWeightsSize size;
    size.w = in * out;
    size.u = out * out;
    size.b_i = out;
    size.b_h = out;
    size.sum = size.w + size.u + size.b_h + size.b_i;
    return size;
}

RNNConfig RNNConfigCreate(
    int input_feature_channels,
    int output_feature_channels,
    bool return_sequences,
    int timesteps,
    bool v2,
    ActivationFunction activation
) {
    RNNConfig result;
    result.base = RecurrentConfigCreate(input_feature_channels, output_feature_channels, return_sequences, timesteps);
    result.v2 = v2;
    result.activation = activation;
    return result;
}

void rnn_training_data_destroy(RNNTrainingData *data) {
    free(data->computation_buffer);
    recurrent_gradient_destroy(data->current_weights_gradient);
    free(data);
}

void rnn_inference_data_destroy(RNNInferenceData *data) {
    free(data->computation_buffer);
    free(data);
}

RNNInferenceData *rnn_inference_data_create(RNNConfig config) {
    RNNInferenceData *data = malloc(sizeof(RNNInferenceData));
    data->computation_buffer = f_malloc(3 * config.base.output_feature_channels);
    return data;
}

RNNTrainingData *rnn_training_data_create(RNNConfig config, RecurrentTrainingConfig training_config) {
    RNNTrainingData *data = malloc(sizeof(RNNTrainingData));
    int out = config.base.output_feature_channels;
    int batch = training_config.mini_batch_size;
    int input_size = batch * config.base.input_feature_channels * config.base.timesteps;
    int output_size = batch * out * config.base.timesteps;
    data->config = training_config;
    //input size 0 because we need weights grad only
    data->current_weights_gradient = recurrent_gradient_create(
            rnn_weights_size_from_config(config), 0);
    data->input = f_malloc(input_size + 2 * out + 3 * output_size);
    data->output = data->input + input_size;
    data->gate = data->output + output_size;
    data->dH = data->gate + output_size;
    data->computation_buffer = data->dH + output_size;
    return data;
}

RecurrentWeights *RNNGetWeights(RNN filter) {
    return filter->weights;
}

RNN rnn_create(RNNConfig config) {
    RNN filter = malloc(sizeof(struct RNNStruct));
    filter->config = config;
    filter->weights = recurrent_weights_create(rnn_weights_size_from_config(config));
    filter->h = f_malloc(config.base.output_feature_channels);
    filter->training_data = NULL;
    filter->inference_data = NULL;
    return filter;
}

RNN RNNCreateForInference(RNNConfig config) {
    RNN rnn = rnn_create(config);
    rnn->inference_data = rnn_inference_data_create(config);
    return rnn;
}

RNN RNNCreateForTraining(RNNConfig config, RecurrentTrainingConfig training_config) {
    RNN rnn = rnn_create(config);
    rnn->training_data = rnn_training_data_create(config, training_config);
    return rnn;
}

RNNGradient *RNNGradientCreate(RNNConfig config, RNNTrainingConfig training_config) {
    return recurrent_gradient_create(
        rnn_weights_size_from_config(config),
        training_config.mini_batch_size *
        config.base.timesteps * config.base.input_feature_channels
    );
}

void RNNDestroy(RNN filter) {
    recurrent_weights_destroy(filter->weights);
    free(filter->h);
    if (filter->training_data != NULL) {
        rnn_training_data_destroy(filter->training_data);
    }
    if (filter->inference_data != NULL) {
        rnn_inference_data_destroy(filter->inference_data);
    }
}

void RNNCellForward(
        RecurrentWeights *weights,
        ActivationFunction activation,
        bool v2,
        int in,
        int out,
        const float *input,
        float *h_prev,
        float *h,
        float *buffer,
        float *gate
) {
    float *x_W = buffer;
    op_mat_mul(input, weights->W, x_W, 1, out, in);
    op_vec_add(x_W, weights->b_i, x_W, out);
    float *h_U = x_W + out;
    op_mat_mul(h_prev, weights->U, h_U, 1, out, out);
    if (v2) {
        op_vec_add(h_U, weights->b_h, h_U, out);
    }
    op_vec_add(h_U, x_W, gate, out);
    ActivationFunctionApply(activation, gate, h);
}

typedef struct {
    float *gate;
    float *h_t;
    float *x_t;
    float *h_t_prev;
} CellBackwardCache;

typedef struct {
    float *d_h_t_prev;
    float *d_x_t;
    float *d_W_t;
    float *d_U_t;
    float *d_bi_t;
    float *d_bh_t;
} CellBackwardGradients;

void RNNCellBackward(
        RecurrentWeights *weights,
        ActivationFunction activation,
        int in,
        int out,
        float *d_H,
        CellBackwardCache cache,
        CellBackwardGradients gradients
) {
    float d_gate[out];
    /*
     * d_gate = d_activation_fn * d_h
     * */
    ActivationFunctionCalculateGradient(activation, cache.gate, cache.h_t, d_H, d_gate);
    /*
     * db = d_gate
     * */
    f_copy(gradients.d_bi_t, d_gate, out);
    f_copy(gradients.d_bh_t, d_gate, out);
    /*
     * d_x = W * d_gate
     * d_h_t_prev = U * d_gate
     * */
    op_mat_mul(weights->W, d_gate, gradients.d_x_t, in, 1, out);
    op_mat_mul(weights->U, d_gate, gradients.d_h_t_prev, out, 1, out);
    /*
     Final backward step:
        d_w_t = d_gate * x_t
        d_u_t = d_gate * h_t-1
        d_b = d_gates
    */
    op_mat_mul(cache.x_t, d_gate, gradients.d_W_t, in, out, 1);
    if (cache.h_t_prev) {
        op_mat_mul(cache.h_t_prev, d_gate, gradients.d_U_t, out, out, 1);
    } else {
        f_zero(gradients.d_U_t, out * out);
    }
}

int RNNApplyInference(RNN filter, const float *input, float *output) {
    if (filter->training_data != NULL) {
        return -1;
    }
    int out = filter->config.base.output_feature_channels;
    int in = filter->config.base.input_feature_channels;
    for (int i = 0; i < filter->config.base.timesteps; ++i) {
        int output_offset = filter->config.base.return_sequences ? i * out : 0;
        RNNCellForward(
                filter->weights,
                filter->config.activation,
                filter->config.v2,
                in, out, input + i * in,
                filter->h,
                output + i * output_offset,
                filter->inference_data->computation_buffer,
                filter->inference_data->computation_buffer + 2 * out
        );
        f_copy(filter->h, output + output_offset, out);
    }
    return 0;
}

int RNNApplyTrainingBatch(RNN filter, const float *input, float *output) {
    if (filter->training_data == NULL) {
        return -1;
    }

    int out = filter->config.base.output_feature_channels;
    int in = filter->config.base.input_feature_channels;
    int batch = filter->training_data->config.mini_batch_size;
    int ts = filter->config.base.timesteps;

    int input_buffer_size = batch * ts * in;
    f_copy(filter->training_data->input, input, input_buffer_size);

    for (int b = 0; b < batch; ++b) {
        f_zero(filter->h, out);
        for (int i = 0; i < filter->config.base.timesteps; ++i) {

            const float *x_t = input + i * in + b * ts * in;
            float *h_t = filter->training_data->output + out * i + b * ts * out;
            float *gate = filter->training_data->gate + out * i + b * ts * out;
            float *h_t_prev = filter->h;

            RNNCellForward(
                    filter->weights,
                    filter->config.activation,
                    filter->config.v2,
                    in, out, x_t,
                    h_t_prev,
                    h_t,
                    filter->training_data->computation_buffer,
                    gate
            );

            f_copy(filter->h, h_t, out);
        }
    }
    if (filter->config.base.return_sequences) {
        f_copy(output, filter->training_data->output, batch * ts * out);
    } else {
        for (int b = 0; b < batch; ++b) {
            int offset = ((ts - 1) * out) + b * ts * out;
            f_copy(output + b * out, filter->training_data->output + offset, out);
        }
    }
    return 0;
}

void RNNCalculateGradient(RNN filter, RNNGradient *gradient, float *d_out) {
    if (filter->training_data == NULL) {
        return;
    }

    int batch = filter->training_data->config.mini_batch_size;
    int ts = filter->config.base.timesteps;
    int in = filter->config.base.input_feature_channels;
    int out = filter->config.base.output_feature_channels;

    float *h = filter->training_data->output;
    float *x = filter->training_data->input;
    float *gate = filter->training_data->gate;

    float *dh = filter->training_data->dH;

    for (int b = 0; b < batch; ++b) {
        for (int t = ts - 1; t >= 0; --t) {
            size_t t_out_offset = t * out + b * ts * out;

            CellBackwardCache cache;

            cache.h_t_prev = t == 0 ? NULL : h + (t_out_offset - out);
            cache.x_t = x + t * in + b * ts * in;
            cache.h_t = h + t_out_offset;
            cache.gate = gate + t_out_offset;

            CellBackwardGradients current_gradients;

            current_gradients.d_W_t = filter->training_data->current_weights_gradient->d_W;
            current_gradients.d_U_t = filter->training_data->current_weights_gradient->d_U;
            current_gradients.d_bi_t = filter->training_data->current_weights_gradient->d_b_i;
            current_gradients.d_bh_t = filter->training_data->current_weights_gradient->d_b_h;

            current_gradients.d_h_t_prev = dh + (b * out);
            current_gradients.d_x_t = gradient->d_X + (t * in + b * ts * in);


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


            RNNCellBackward(filter->weights, filter->config.activation, in, out, d_h_t, cache,
                            current_gradients);

            recurrent_gradient_sum(filter->training_data->current_weights_gradient, gradient, rnn_weights_size_from_config(filter->config));
        }
    }
}




