//
//  File.c
//  
//
//  Created by Alex on 01.10.2020.
//

#include "lstm.h"
#include "stdlib.h"
#include "string.h"
#include "activation_default.h"
#include "operations.h"
#include "debug.h"

LSTMActivations LSTMActivationsCreate(ActivationFunction inputGateActivation, ActivationFunction forgetGateActivation, ActivationFunction candidateGateActivation, ActivationFunction outputGateActivation, ActivationFunction outputActivation){
    LSTMActivations activations;
    activations.candidateGateActivation = candidateGateActivation;
    activations.forgetGateActivation = forgetGateActivation;
    activations.inputGateActivation = inputGateActivation;
    activations.outputActivation = outputActivation;
    activations.outputGateActivation = outputGateActivation;
    return activations;
}

typedef struct {
    LSTMTrainingConfig config;
    float *input;
    float *zifgo;
    float *state;
    float *output;
    float *dH;
    float *dC;

} LSTMTrainingData;

void LSTMTrainingDataDestroy(LSTMTrainingData *data){
    free(data->input);
    free(data);
}

LSTMTrainingData* LSTMTrainingDataCreate(LSTMConfig config,  LSTMTrainingConfig trainingConfig){
    LSTMTrainingData *trainingData = malloc(sizeof(LSTMTrainingData));

    trainingData->config = trainingConfig;

    int in = config.inputFeatureChannels;
    int out = config.outputFeatureChannels;
    int ts = config.timesteps;
    int batch = trainingConfig.mini_batch_size;
    /*
     *input -> *zifgo(8 out) -> *state(out) -> *output(out) -> *dHt(out) -> *dCt(out)
    */
    int input = batch * in * ts;
    int trainingCacheSize = (input + batch * ts * 10 * out + 2 * batch * out) * sizeof(float);

    trainingData->input = malloc(trainingCacheSize);
    trainingData->zifgo = trainingData->input + input;
    trainingData->state = trainingData->zifgo + 8 * batch * ts * out;
    trainingData->output = trainingData->state + batch * ts * out;
    trainingData->dH = trainingData->output + batch * ts * out;
    trainingData->dC = trainingData->dH + batch * out;

    memset(trainingData->input, 0, trainingCacheSize);
}

struct LSTMFilterStruct {
    LSTMConfig config;
    float *forwardComputationBuffer;
    float *state;
    float *output;
    LSTMTrainingData* trainingData;
    LSTMWeights* weights;
};

LSTMActivations LSTMActivationsCreateDefault(int size){
    return LSTMActivationsCreate(
        ActivationFunctionCreateSigmoid(size),
        ActivationFunctionCreateSigmoid(size),
        ActivationFunctionCreateTanh(size),
        ActivationFunctionCreateSigmoid(size),
        ActivationFunctionCreateTanh(size)
   );
}

void LSTMActivationsDestroy(LSTMActivations activations){
    ActivationFunctionDestroy(activations.inputGateActivation);
    ActivationFunctionDestroy(activations.forgetGateActivation);
    ActivationFunctionDestroy(activations.candidateGateActivation);
    ActivationFunctionDestroy(activations.outputGateActivation);
    ActivationFunctionDestroy(activations.outputActivation);
}

LSTMWeights* LSTMFilterGetWeights(LSTMFilter filter){
    return filter->weights;
}

LSTMConfig LSTMConfigCreate(int inputFeatureChannels, int outputFeatureChannels, bool v2, bool returnSequences, int timesteps, LSTMActivations activations){
    LSTMConfig config;
    config.inputFeatureChannels = inputFeatureChannels;
    config.outputFeatureChannels = outputFeatureChannels;
    config.v2 = v2;
    config.returnSequences = returnSequences;
    config.activations = activations;
    config.timesteps = timesteps;
    return config;
}

typedef struct {
    int w;
    int u;
    int b_i;
    int b_h;
    int buffer;
} LSTMWeightsSize;

LSTMWeightsSize LSTMWeightsSizeFromConfig(LSTMConfig config){
    int in = config.inputFeatureChannels;
    int out = config.outputFeatureChannels;
    LSTMWeightsSize size;
    size.w = 4 * in * out;
    size.u = 4 * out * out;
    size.b_i = 4 * out;
    size.b_h = 4 * out;
    size.buffer = (size.w + size.u + size.b_h + size.b_i) * sizeof(float);
    return size;
}

LSTMFilter LSTMFilterCreate(LSTMConfig config){
    LSTMFilter filter = malloc(sizeof(struct LSTMFilterStruct));

    int out = config.outputFeatureChannels;

    filter->config = config;

    filter->weights = malloc(sizeof(LSTMWeights));

    int outSize = out * sizeof(float);
    filter->state = malloc(outSize);
    filter->output = malloc(outSize);
    memset(filter->state, 0, outSize);
    memset(filter->output, 0, outSize);

    LSTMWeightsSize sizes = LSTMWeightsSizeFromConfig(config);
    float *buffer = malloc(sizes.buffer);
    memset(buffer, 0, sizes.buffer);
    filter->weights->W = buffer;
    filter->weights->U = filter->weights->W + sizes.w;
    filter->weights->b_i = filter->weights->U + sizes.u;
    filter->weights->b_h = filter->weights->b_i + sizes.b_i;

    return filter;
}

LSTMFilter LSTMFilterCreateForInference(LSTMConfig config){
    LSTMFilter filter = LSTMFilterCreate(config);
    int computationBufferSize = 11 * config.outputFeatureChannels * sizeof(float);
    filter->forwardComputationBuffer = malloc(computationBufferSize);
    memset(filter->forwardComputationBuffer, 0, computationBufferSize);
    return filter;
}

LSTMTrainingConfig LSTMTrainingConfigCreate(int mini_batch_size){
    LSTMTrainingConfig config;
    config.mini_batch_size = mini_batch_size;
    return config;
}

LSTMFilter LSTMFilterCreateForTraining(LSTMConfig config, LSTMTrainingConfig trainingConfig){
    LSTMFilter filter = LSTMFilterCreate(config);

    int computationBufferSize = 3 * config.outputFeatureChannels * sizeof(float);
    filter->forwardComputationBuffer = malloc(computationBufferSize);
    memset(filter->forwardComputationBuffer, 0, computationBufferSize);

    filter->trainingData = LSTMTrainingDataCreate(config, trainingConfig);

    return filter;
}

void LSTMFilterDestroy(LSTMFilter filter) {
    free(filter->weights->W);
    free(filter->forwardComputationBuffer);
    free(filter->weights);
    free(filter->output);
    free(filter->state);
    if (filter->trainingData){
        LSTMTrainingDataDestroy(filter->trainingData);
    }
    free(filter);
}

void LSTMCellForward(
     LSTMWeights *weights,
     LSTMActivations activations,
     int in,
     int out,
     const float *input,
     float *c_prev,
     float *h_prev,
     float *c,
     float *h,
     float *zifgo,
     float *buffer
 ){
    float *Z = zifgo;
    // Z = input * W (WI, WF, WG, WO in row) + h * U (UI, UF, UG, UO in row) + bias(BI, BF, BG, BO)
    MatMul(input, weights->W, Z, 1, 4 * out, in, 0.0);
    //out = x * W + b_i
    VectorAdd(Z, weights->b_i, Z, 4 * out);
    // in_U = h_t * U
    MatMul(h_prev, weights->U, Z, 1, out * 4, out, 1.0);
    // input Gate =  recurrent_activation(Z[0: out])
    // default sigmoid
    float* i = Z + 4 * out;
    ActivationFunctionApply(activations.inputGateActivation, Z, i);
    // forget Gate =  input_activation(Z[out: 2 * out])
    // default sigmoid
    float* f = i + out;
    ActivationFunctionApply(activations.forgetGateActivation, Z + out, f);
    // candidate Gate = activation(Z[2 * out: 3 *out])
    // default tanh
    float *g = f + out;
    ActivationFunctionApply(activations.candidateGateActivation, Z + 2 * out, g);
    // output Gate = recurrent_activation(Z[3 * out: 4 *out])
    // default sigmoid
    float *o = g + out;
    ActivationFunctionApply(activations.outputGateActivation, Z + 3 * out, o);
    // ig = i * g
    float* i_g = buffer;
    VectorMul(i, g, i_g, out);
    //f_cpr = f * c_pr
    float*  f_c_pr = i_g + out;
    VectorMul(f, c_prev, f_c_pr, out);
    //c = fc_pr + i_g
    VectorAdd(f_c_pr, i_g, c, out);
    // H = o * tanh(c)
    float* c_tanh = f_c_pr + out;
    ActivationFunctionApply(activations.outputActivation, c, c_tanh);
    VectorMul(o, c_tanh, h, out);
}

int LSTMFilterApply(LSTMFilter filter, const float *input, float* output){
    int out = filter->config.outputFeatureChannels;
    int in = filter->config.inputFeatureChannels;
    for (int i = 0; i < filter->config.timesteps; ++i){
        float state[out];
        int outputIndex = filter->config.returnSequences ? i * out : 0;
        LSTMCellForward(filter->weights, filter->config.activations, in, out, input + i * in, filter->state, filter->output, state, output + outputIndex, filter->forwardComputationBuffer, filter->forwardComputationBuffer + 8);
        memcpy(filter->output, output + outputIndex, out * sizeof(float));
        memcpy(filter->state, state, out * sizeof(float));
    }
    return 0;
}

void LSTMFilterZeroState(LSTMFilter filter){
    int size = filter->config.outputFeatureChannels * sizeof(float);
    memset(filter->state, 0, size);
    memset(filter->output, 0, size);
}

typedef struct {
    float *zifgo;
    float *x_t;
    float *c_t;
    float *h_t_prev;
    float *c_t_prev;
} CellBackwardCache;

typedef struct {
    float *d_h_t_prev;
    float *d_x_t;
    float *d_W_t;
    float *d_U_t;
    float *d_bi_t;
    float *d_bh_t;
    float *d_c_t_prev;
} CellBackwardGradients;

void LSTMCellBackward(
      LSTMWeights *weigths,
      LSTMActivations activations,
      int in,
      int out,
      const float *d_h_t,
      const float *d_c_t_init,
      CellBackwardCache cache,
      CellBackwardGradients gradients,
      float *buffer
){
    // convenient cache;
    float *z_i_t = cache.zifgo;
    float *z_f_t = z_i_t + out;
    float *z_g_t = z_f_t + out;
    float *z_o_t = z_g_t + out;

    float *i_t = z_o_t + out;
    float *f_t = i_t + out;
    float *g_t = f_t + out;
    float *o_t = g_t + out;

    float *dgates = buffer;

    float *d_i_t = dgates;
    float *d_f_t = d_i_t + out;
    float *d_g_t = d_f_t + out;
    float *d_o_t = d_g_t + out;
    /*
     Backward step 1.
        FWD: h_t = tanh(c_t) * OutputGate(t);

        d_o_t = d_h_t * tanh(c_t);
        d_c_t = d_h_t * o_t * (1 - tanh^2(c_t))); //From current formula
        d_c_t += d_ct (from previous calculation) (d_c_t+1 * f_t+1) in step 2 you will see "Previous state"
     */
    // d_o_t
    ActivationFunctionApply(activations.outputActivation, cache.c_t, d_o_t);
    float *d_out_gate_act = d_o_t + out;
    ActivationFunctionApplyDerivative(activations.outputGateActivation, z_o_t, o_t, d_out_gate_act);
    VectorMul(d_o_t, d_out_gate_act, d_o_t, out);
    VectorMul(d_h_t, d_o_t, d_o_t, out);
    // d_c_t
    float *d_c_t = d_out_gate_act + out;
    ActivationFunctionApplyDerivative(activations.outputActivation, cache.c_t, NULL, d_c_t);
    VectorMul(d_c_t, o_t, d_c_t, out);
    VectorMul(d_h_t, d_c_t, d_c_t, out);
    if (d_c_t_init != NULL)
        VectorAdd(d_c_t, d_c_t_init, d_c_t, out);
    /*
     Backward step 2
        FWD: c_t = i_t * g_t + f_t * c_t-1

        For calculating d_gate we need to use derivatives associated with W, U, h_t-1 and x_t
        d_sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) <=>  d_activation_sigmoid(z_t) = a_t * (1 - a_t);
        d_tanh(x) = (1 - tanh(x)^2) <=> d_activation_tanh(z_t) = 1 - (a_t ^ 2);

        gates: i, f, g, o (input, forget, candidate, output)
        Z_t = [i_t, f_t, g_t, o_t];
        dZt = [dZit, dZft, dZgt, dZot];

        z_(gate)_t = x_t * W_(gate) + h_t-1 * U_(gate) + b_(gate);
        a_gate_t = (gate)_activation(z_(gate)_t);

        Input gate:
        d_a_i_t = d_c_t * g_t;
        d_z_i_t = d_a_i_t * d_input_activation(z_i_t);
        for default sigmoid => d_z_i_t = d_c_t * g_t * a_i_t * (1 - a_i_t);
     */
    ActivationFunctionApplyDerivative(activations.inputGateActivation, z_i_t, i_t, d_i_t);
    VectorMul(d_c_t, d_i_t, d_i_t, out);
    VectorMul(g_t, d_i_t, d_i_t, out);
    /*
        Forget gate:
        d_a_f_t = d_c_t * c_t-1;
        d_z_f_t = d_a_f_t * d_forget_activation(z_f_t);
        for default sigmoid => d_z_f_t = d_c_t * c_t-1 * a_f_t * (1 - a_f_t);
     */
    ActivationFunctionApplyDerivative(activations.forgetGateActivation, z_f_t, f_t, d_f_t);
    VectorMul(d_c_t, d_f_t, d_f_t, out);
    if (cache.c_t_prev == NULL){
        memset(d_f_t, 0, out * sizeof(float));
    } else {
        VectorMul(cache.c_t_prev, d_f_t, d_f_t, out);
    }
    /*
        Candidate gate:
        d_a_g_t = d_c_t * i_t;
        d_z_g_t = d_a_g_t * d_candidate_activation(z_g_t);
        for default tanh => d_z_g_t = d_c_t * i_t * (1 - (g_t^2));
    */
    ActivationFunctionApplyDerivative(activations.candidateGateActivation, z_g_t, g_t, d_g_t);
    VectorMul(d_c_t, d_g_t, d_g_t, out);
    VectorMul(i_t, d_g_t, d_g_t, out);
    /*
       Previous state:
       d_c_t-1 = d_c_t * f_t
     */
    VectorMul(d_c_t, f_t, gradients.d_c_t_prev, out);
    /*
     Backward step 3:
        d_x_t = dgates * WT;
        d_h_t-1 = dgates * UT;
     */
    MatMul3(weigths->W, dgates, false, false, gradients.d_x_t, in, 1, 4 * out, 0.0);
    MatMul3(weigths->U, dgates, false, false, gradients.d_h_t_prev, out, 1, 4 * out, 0.0);
    /*
     Final backward step:
        d_w_t = d_gates * x_t
        d_u_t = d_gates * h_t-1
        d_b = d_gates
    */
    MatMul3(cache.x_t, dgates, false, false, gradients.d_W_t, in, 4 * out, 1, 0.0);
    if (cache.h_t_prev){
        MatMul3(cache.h_t_prev, dgates, false, false, gradients.d_U_t, out, 4 * out, 1, 0.0);

    } else {
        memset(gradients.d_U_t, 0, 4 * out * out * sizeof(float));
    }
    memcpy(gradients.d_bi_t, dgates, 4 * out * sizeof(float));
    memcpy(gradients.d_bh_t, dgates, 4 * out * sizeof(float));
}

int LSTMFilterApplyTrainingBatch(LSTMFilter filter, const float *input, float* output){
    if (filter->trainingData == NULL){
        return -1;
    }
    int out = filter->config.outputFeatureChannels;
    int in = filter->config.inputFeatureChannels;
    int batch = filter->trainingData->config.mini_batch_size;
    int ts = filter->config.timesteps;

    int inputBufferSize = batch * ts * in * sizeof(float);
    memcpy(filter->trainingData->input, input, inputBufferSize);

    for (int b = 0; b < batch; ++b){
        LSTMFilterZeroState(filter);
        for (int i = 0; i < filter->config.timesteps; ++i){

            const float *x_t = input + i * in + b * ts * in;

            float *c_t = filter->trainingData->state + out * i + b * ts * out;
            float *h_t = filter->trainingData->output + out * i + b * ts * out;
            float *zifgo = filter->trainingData->zifgo + i * 8 * out + 8 * b * ts * out;

            float *c_t_prev = filter->state;
            float *h_t_prev = filter->output;

            LSTMCellForward(filter->weights, filter->config.activations, in, out, x_t, c_t_prev, h_t_prev, c_t, h_t, zifgo , filter->forwardComputationBuffer);

            memcpy(filter->output, h_t, out * sizeof(float));
            memcpy(filter->state, c_t, out * sizeof(float));
        }
    }
    if (filter->config.returnSequences){
        memcpy(output, filter->trainingData->output, batch * ts * out * sizeof(float));
    } else {
        for (int b = 0; b < batch; ++b){
            int offset = ((ts - 1) * out) + b * ts * out;
            memcpy(output + b * out, filter->trainingData->output + offset, out * sizeof(float));
        }
    }
    return 0;
}

LSTMGradient * LSTMGradientCreate(LSTMConfig config, LSTMTrainingConfig trainingConfig) {
    LSTMGradient * gradients = malloc(sizeof(LSTMGradient));
    LSTMWeightsSize sizes = LSTMWeightsSizeFromConfig(config);
    int batch = trainingConfig.mini_batch_size;
    int buff_size = sizes.buffer * batch + batch * config.timesteps * config.inputFeatureChannels * sizeof(float);
    gradients->d_W = malloc(buff_size);
    gradients->d_U = gradients->d_W + batch * sizes.w;
    gradients->d_b_i = gradients->d_U + batch * sizes.u;
    gradients->d_b_h = gradients->d_b_i + batch * sizes.b_i;
    gradients->d_X = gradients->d_b_h + batch * sizes.b_h;
    memset(gradients->d_W, 0, buff_size);
    return gradients;
}

void LSTMGradientsDestroy(LSTMGradient *gradient) {
    free(gradient->d_W);
    free(gradient);
}

void LSTMFilterCalculateGradient(LSTMFilter filter, LSTMGradient *gradient, float *dout) {
    if (filter->trainingData == NULL){
        return;
    }

    int batch = filter->trainingData->config.mini_batch_size;
    int ts = filter->config.timesteps;
    int in = filter->config.inputFeatureChannels;
    int out = filter->config.outputFeatureChannels;

    float *c = filter->trainingData->state;
    float *h = filter->trainingData->output;
    float *x = filter->trainingData->input;
    float *zifgo = filter->trainingData->zifgo;

    LSTMWeightsSize sizes = LSTMWeightsSizeFromConfig(filter->config);

    int buffer_size = sizes.buffer * batch + batch * in * ts * sizeof(float);

    float *dW = malloc(buffer_size);
    float *dU = dW + batch * sizes.w;
    float *d_bi = dU + batch * sizes.u;
    float *d_bh = d_bi + batch * sizes.b_i;
    float *dx = d_bh + batch * sizes.b_h;

    float *dc = filter->trainingData->dC;
    float *dh = filter->trainingData->dH;

    memset(dW, 0, sizes.buffer * batch);

    int computation_buffer_size = 6 * out * sizeof(float);
    float *computation_buffer = (float *) malloc(computation_buffer_size);
    memset(computation_buffer, 0, computation_buffer_size);

    S_LOOP_START(batch, b)
        for(int t = ts - 1; t >= 0; --t){
            size_t tOutOffset = t * out + b * ts * out;

            CellBackwardCache cache;

            cache.c_t = c + tOutOffset;
            cache.c_t_prev = t == 0 ? NULL : c + (tOutOffset - out);
            cache.h_t_prev = t == 0 ? NULL : h + (tOutOffset - out);
            cache.x_t = x + t * in + b * ts * in;
            cache.zifgo = zifgo + (8 * tOutOffset);

            CellBackwardGradients currentGradients;

            currentGradients.d_W_t = dW + b * sizes.w;
            currentGradients.d_U_t = dU + b * sizes.u;
            currentGradients.d_bi_t = d_bi + b * sizes.b_i;
            currentGradients.d_bh_t = d_bh + b * sizes.b_h;
            currentGradients.d_c_t_prev = dc + (b * out);
            currentGradients.d_h_t_prev = dh + (b * out);
            currentGradients.d_x_t = dx + (t * in + b * ts * in);

            float *d_c_t_init = t == ts - 1 ? NULL : dc + (b * out);
            float *d_h_t_init = t == ts - 1 ? NULL : dh + (b * out);

            bool seq = filter->config.returnSequences;

            float d_out_t[out];
            memset(d_out_t, 0, out * sizeof(float));
            if (seq){
                memcpy(d_out_t, dout + b * ts * out + t * out, out * sizeof(float));
            } else if (t == ts - 1){
                memcpy(d_out_t, dout + b * out, out * sizeof(float));
            }

            float d_h_t[out];
            memset(d_h_t, 0, out * sizeof(float));

            VectorAdd(d_h_t_init == NULL ? d_h_t : d_h_t_init, d_out_t, d_h_t, out);

            LSTMCellBackward(filter->weights, filter->config.activations, in, out, d_h_t, d_c_t_init, cache, currentGradients, computation_buffer);
            memset(computation_buffer, 0, computation_buffer_size);
            
            VectorAdd(gradient->d_W + b * sizes.w, currentGradients.d_W_t, gradient->d_W + b * sizes.w, sizes.w);
            VectorAdd(gradient->d_U + b * sizes.u, currentGradients.d_U_t, gradient->d_U + b * sizes.u, sizes.u);
            VectorAdd(gradient->d_b_i + b * sizes.b_i, currentGradients.d_bi_t, gradient->d_b_i + b * sizes.b_i, sizes.b_i);
            VectorAdd(gradient->d_b_h + b * sizes.b_h, currentGradients.d_bh_t, gradient->d_b_h + b * sizes.b_h, sizes.b_h);
        }
    S_LOOP_END

    free(dW);
}
