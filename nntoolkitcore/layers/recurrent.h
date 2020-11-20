//
// Created by Alex on 20.11.2020.
//

#ifndef recurrent_h
#define recurrent_h

typedef struct {
    float *W;
    float *U;
    float *b_i;
    float *b_h;
} RecurrentWeights;

typedef struct {
    int mini_batch_size;
} RecurrentTrainingConfig;

typedef struct {
    float * d_W;
    float * d_U;
    float * d_b_i;
    float * d_b_h;
    float * d_X;
} RecurrentGradient;

typedef struct {
    int w;
    int u;
    int b_i;
    int b_h;
    int sum;
} RecurrentWeightsSize;

inline static RecurrentTrainingConfig RecurrentTrainingConfigCreate(int mini_batch_size){
    RecurrentTrainingConfig config;
    config.mini_batch_size = mini_batch_size;
    return config;
}

#endif //recurrent
