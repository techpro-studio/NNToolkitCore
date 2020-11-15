//
//  activation_default_functions.c
//  audio_test
//
//  Created by Alex on 28.09.2020.
//  Copyright © 2020 Alex. All rights reserved.
//

#include "nntoolkitcore/layers/activation_default.h"
#include "nntoolkitcore/core/ops.h"
#include "nntoolkitcore/core/loop.h"
#include "stdlib.h"
#include "string.h"


ActivationFunction create_simple(int size, ActivationFunctionImpl function, ActivationFunctionDerivative derivative) {
    return ActivationFunctionCreate(size, NULL, NULL, function, derivative, NULL);
}

ActivationFunction
create_simple_with_cached(int size, ActivationFunctionImpl function, ActivationFunctionDerivative derivative,
                          ActivationFunctionDerivative cachedDerivative) {
    return ActivationFunctionCreate(size, NULL, NULL, function, derivative, cachedDerivative);
}

//SIGMOID

// 1 / 1 - exp(-x)
void activation_sigmoid(void *implementer, const float *input, float *output, int size) {
    op_vec_neg(input, output, size);
    op_vec_exp(output, output, size);
    op_vec_add_sc(output, 1, output, size);
    op_vec_reciprocal(output, output, size);
}


// sigmoid(x)' = sigmoid(x)* (1 * sigmoid(x));

void activation_sigmoid_cached_derivative(void *implementer, const float *input, const float *d_out, float *output,
                                          int size) {
    op_vec_neg(input, output, size);
    op_vec_add_sc(output, 1, output, size);
    op_vec_mul(input, output, output, size);
    //d_out * d_activation_sigmoid
    op_vec_mul(output, d_out, output, size);
}


void activation_sigmoid_derivative(void *implementer, const float *input, const float *d_out, float *output, int size) {
    activation_sigmoid(implementer, input, output, size);
    activation_sigmoid_cached_derivative(implementer, output, d_out, output, size);
}


ActivationFunction ActivationFunctionCreateSigmoid(int inputSize) {
    return create_simple_with_cached(inputSize, activation_sigmoid, activation_sigmoid_derivative,
                                     activation_sigmoid_cached_derivative);
}



// TANH



void activation_tanh(void *implementer, const float *input, float *output, int size) {
    op_vec_tanh(input, output, size);
}

void
activation_tanh_cached_derivative(void *implementer, const float *input, const float *d_out, float *output, int size) {
    op_vec_mul(input, input, output, size);
    op_vec_neg(output, output, size);
    op_vec_add_sc(output, 1, output, size);
    //d_out * d_activation_sigmoid
    op_vec_mul(output, d_out, output, size);

}

void activation_tanh_derivative(void *implementer, const float *input, const float *d_out, float *output, int size) {
    activation_tanh(implementer, input, output, size);
    activation_tanh_cached_derivative(implementer, output, d_out, output, size);
}


ActivationFunction ActivationFunctionCreateTanh(int inputSize) {
    return create_simple_with_cached(inputSize, activation_tanh, activation_tanh_derivative,
                                     activation_tanh_cached_derivative);
}

//Identity


void
activation_identity_derivative(void *implementer, const float *input, const float *d_out, float *output, int size) {
    memcpy(output, d_out, size * sizeof(float));
}

void activation_identity(void *implementer, const float *input, float *output, int size) {
    if (input == output) {
        return;
    }
    memcpy(output, input, size * sizeof(float));
}

ActivationFunction ActivationFunctionCreateIdentity(int inputSize) {
    return create_simple(inputSize, activation_identity, activation_identity_derivative);
}


// ReLU



typedef struct {
    float a;
} ReLUImplementer;

void activation_relu_derivative(void *implementer, const float *input, const float *d_out, float *output, int size) {
    op_vec_clamp(input, output, 0, 1, size);
    op_vec_mul(output, d_out, output, size);
}

void activation_relu(void *implementer, const float *input, float *output, int size) {
    ReLUImplementer *impl = (ReLUImplementer *) implementer;
    op_vec_max_sc(input, 0, output, size);
    if (impl->a != 1.0) {
        op_vec_mul_sc(output, impl->a, output, size);
    }
}

void relu_implementer_destroy(void *ptr) {
    free(ptr);
}

ActivationFunction ActivationFunctionCreateReLU(int inputSize, float a) {
    ReLUImplementer *implementer = malloc(sizeof(ReLUImplementer));
    implementer->a = a;
    return ActivationFunctionCreate(inputSize, relu_implementer_destroy, implementer, activation_relu,
                                    activation_relu_derivative, NULL);
}



// HARD SIGMOID



//typedef struct {
//    op_vec_clamp_fn clamp_fn;
//} HardSigmoidImplementer;
//
//void activation_hard_sigmoid(void *implementer, const float *input, float *output, int size){
//    HardSigmoidImplementer* impl = (HardSigmoidImplementer *)implementer;
//    op_vec_mul_sc(input, 0.2f, output, size);
//    op_vec_add_sc(output, 0.5f, output, size);
//    impl->clamp_fn(output, output, 0, 1, size);
//}
//
//void activation_hard_sigmoid_derivative(void *implementer, const float *input, float *output, int size) {
//#warning implement this;
//    memcpy(output, input, size * sizeof(float));
//}
//
//void HardSigmoidImplementerDestroy(void * ptr){
//    free(ptr);
//}
//
//ActivationFunction ActivationFunctionCreateHardSigmoid(int input_size){
//    HardSigmoidImplementer* implementer = malloc(sizeof(HardSigmoidImplementer));
//    implementer->clamp_fn = op_vec_clamp_get_optimized(input_size);
//    return ActivationFunctionCreate(input_size, HardSigmoidImplementerDestroy, implementer, activation_hard_sigmoid, activation_hard_sigmoid_derivative, NULL);
//}


// SOFTMAX


typedef struct {
    int vector_size;
} SoftmaxImplementer;


void softmax(const float *input, float *output, int vector_size) {
    op_vec_exp(input, output, vector_size);
    float sum = 0.0f;
    op_vec_sum(output, &sum, vector_size);
    op_vec_div_sc(output, sum, output, vector_size);
}


void activation_softmax(void *implementer, const float *input, float *output, int size) {
    int v_size = ((SoftmaxImplementer *) implementer)->vector_size;
    if (size == 1) {
        softmax(input, output, v_size);
        return;
    }
    P_LOOP_START(size, index)
        int offset = index * v_size;
        softmax(input + offset, output + offset, v_size);
    P_LOOP_END
}

void activation_softmax_derivative_cached(void *implementer, const float *input, const float *d_out, float *output, int size) {
    int v_size = ((SoftmaxImplementer *) implementer)->vector_size;
    P_LOOP_START(size, index)
        int offset = index * v_size;
        const float *in = input + offset;
        float *out = output + offset;
        float m_derivative[v_size * v_size];
        for (int i = 0; i < v_size; ++i) {
            for (int j = 0; j < v_size; ++j) {
                m_derivative[i * v_size + j] = i == j ?
                                               in[i] * (1 - in[i]) :
                                               -1 * in[i] * in[j];
            }
        }
        op_mat_mul(d_out, m_derivative, out, 1, v_size, v_size);
    P_LOOP_END
}

void activation_softmax_derivative(void *implementer, const float *input, const float *d_out, float *output, int size) {
    activation_softmax(implementer, input, output, size);
    activation_softmax_derivative_cached(implementer, output, d_out,  output, size);
}


void softmax_implementer_destroy(void *ptr) {
    free(ptr);
}

ActivationFunction ActivationFunctionCreateSoftmax(int inputSize, int vectorSize) {
    SoftmaxImplementer *implementer = malloc(sizeof(SoftmaxImplementer));
    implementer->vector_size = vectorSize;
    return ActivationFunctionCreate(inputSize, softmax_implementer_destroy, implementer, activation_softmax,
                                    activation_softmax_derivative, activation_softmax_derivative_cached);
}







