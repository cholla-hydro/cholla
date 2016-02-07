#include<cuda.h>
#include"global.h"
#include"global_cuda.h"

__global__ void test_function()
{
  Real tid;

  tid = threadIdx.x;
  tid += 1;

  return;
}


