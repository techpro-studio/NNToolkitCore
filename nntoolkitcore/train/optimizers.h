//
// Created by Alex on 02.11.2020.
//

#ifndef optimizers_h
#define optimizers_h

#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    float learning_rate;
} SGD;

int sgd_optimize(SGD optimizer, float *gradient, float *weights, int size);

#if defined __cplusplus
}
#endif

#endif //optimizers_h
