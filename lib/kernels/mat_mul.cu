#include <stdio.h>
#include "gnuastro/gpu.h"

__global__ void mat_mul_kernel(int *a, int *b, int *c, int n)
{
    int row = blockIdx.y * n + threadIdx.y;
    int col = blockIdx.x * n + threadIdx.x;
    if (row < n && col < n)
    {
        float product_val = 0;
        for (int k = 0; k < n; k++)
        {
            product_val += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = product_val;
    }

}

void 
gal_gpu_mat_mul(int *a, int *b, int n)
{
    int *c;
    int size = n * n * sizeof(int);
    c = (int *)malloc(size);

    int *device_a, *device_b, *device_c;

    cudaMalloc((void **)&device_a, size);
    cudaMalloc((void **)&device_b, size);
    cudaMalloc((void **)&device_c, size);

    cudaMemcpy(device_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, size, cudaMemcpyHostToDevice);



    dim3 dimBlock(32, 32);
    dim3 dimGrid(ceil(n / 32.0), ceil(n / 32.0));

    mat_mul_kernel<<<dimGrid, dimBlock>>>(device_a, device_b, device_c, n);

    
    cudaMemcpy(c, device_c, size, cudaMemcpyDeviceToHost);
   
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            printf("%d ", c[i*n+j]);
        }
        printf("\n");
    }

    free(c);

}