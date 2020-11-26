//
// Created by Alex on 26.11.2020.
//

#ifndef shared_h
#define shared_h

#if defined __cplusplus
extern "C" {
#endif

typedef struct {
    int mini_batch_size;
} DefaultTrainingConfig;

inline static DefaultTrainingConfig DefaultTrainingConfigCreate(int mini_batch_size) {
    return (DefaultTrainingConfig) {mini_batch_size};
}

#if defined __cplusplus
}
#endif

#endif //shared_h
