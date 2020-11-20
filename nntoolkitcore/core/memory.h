//
// Created by Alex on 20.11.2020.
//

#ifndef memory_h
#define memory_h
#include "string.h"
#include "stdlib.h"

inline void* malloc_zeros(size_t __size) {
    void * buffer = malloc(__size);
    memset(buffer, 0, __size);
    return buffer;
}

#endif //memory_h
