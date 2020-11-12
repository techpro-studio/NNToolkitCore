//
//  Header.h
//  
//
//  Created by Alex on 08.11.2020.
//

#ifndef loop_h
#define loop_h

#include "types.h"


#if APPLE
    #include <dispatch/dispatch.h>
    #define P_LOOP_START(size, var) dispatch_apply(size, DISPATCH_APPLY_AUTO, ^(size_t var) {
    #define P_LOOP_END });
#elif _OPENMP
    #define P_LOOP_START(size var) #pragma omp for       \
        for (var = 0; var < size; ++var) {               \
    #define P_LOOP_END }
#else
    #define P_LOOP_START(size, var) for(int var = 0; var < size; ++var){
    #define P_LOOP_END }
#endif

#endif /* Header_h */
