#pragma once
#ifdef __cplusplus
extern "C" {
#endif

    void kernel_on_gpu();

    void gal_gpu_mat_add(int *a, int *b, int n);

    void gal_gpu_mat_mul(int *a, int *b, int n);
    

#ifdef __cplusplus
}
#endif