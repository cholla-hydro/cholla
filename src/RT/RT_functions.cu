/*! \file RT_functions.cu
 #  \brief Definitions of functions for the RT solver */


#ifdef CUDA
#ifdef RT

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"gpu.hpp"
#include"global.h"
#include"global_cuda.h"
#include"RT_functions.h"

// CPU function that will call the GPU-based RT functions
void rtSolve(void)
{


}

#endif // RT
#endif // CUDA
