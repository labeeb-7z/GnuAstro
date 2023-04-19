#include <stdio.h>
#include <gnuastro/gpu.h>


__global__ void launch_kernel()
{
    printf("Hello World\n");
}


void kernel_on_gpu()
{
    launch_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
}