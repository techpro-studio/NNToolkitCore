//
//  Header.h
//  
//
//  Created by Alex on 08.11.2020.
//

#ifndef loops_h
#define loops_h


#define S_LOOP_START(size, var) for(int var = 0; var < size; ++var){
#define S_LOOP_END }

#if APPLE
    #include <dispatch/dispatch.h>
    #define P_LOOP_START(size, var) dispatch_apply(size, DISPATCH_APPLY_AUTO, ^(size_t var) {
    #define P_LOOP_END });
#else
    #define P_LOOP_START S_LOOP_START
    #define P_LOOP_END S_LOOP_END
#endif

#endif /* Header_h */
