//
//  window.c
//  
//
//  Created by Alex on 04.10.2020.
//

#include "window.h"
#include <math.h>

void hann_family_window(float * vector, int size, float alpha){
    for (int i = 0; i < size; ++i){
        vector[i] = alpha - (1 -  alpha) * cos(2 * M_PI * i / (size - 1));
    }
}

void hamming_window(float * vector, int size){
    hann_family_window(vector, size, 0.54);
}

void hann_window(float *vector, int size){
    hann_family_window(vector, size, 0.5);
}

void blackman_window(float *vector, int size){
    for (int i = 0; i < size; ++i){
        float angle = 2 * M_PI * i / size;
        vector[i] = .42 - .5 * cos(angle) + .08 * cos(2*angle);
    }
}