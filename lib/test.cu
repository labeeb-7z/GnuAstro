#include <stdio.h>
#include "gnuastro/gpu.h"


__global__ void laucn_kernel()
{
    printf("Hello World\n");
}


void kernel_on_gpu()
{
    laucn_kernel<<<1,1>>>();
}