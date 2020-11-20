//
// Created by Alex on 20.11.2020.
//

#ifndef memory_h
#define memory_h
#include "string.h"
#include "stdlib.h"

inline static void* malloc_zeros(size_t __size) {
    void *ptr = malloc(__size);
    memset(ptr, 0, __size);
    return ptr;
}

#endif //memory_h
