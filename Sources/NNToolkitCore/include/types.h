//
// Created by Alex on 10.11.2020.
//

#ifndef complex_h
#define complex_h


typedef struct {
    float real;
    float imag;
} complex_float;

typedef struct {
    float *real_p;
    float *imag_p;
} complex_float_spl;

#endif
