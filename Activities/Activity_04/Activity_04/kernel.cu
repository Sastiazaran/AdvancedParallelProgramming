
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void addKernel(float *a, float *b, float *c, int nfil, int ncol){
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idy * ncol + idx;
    
    if(idy < nfil && idx < ncol) {
        int sum = 0;
        for (int k = 0; k < ncol; k++) {
            sum += a[idy * ncol + k] * b[k * ncol + idx];
        }
        c[index] = sum;
    }
    
}

int main(void)
{
    float* A_h, * B_h, * C_h;
    float* A_d, * B_d, * C_d;
    int nfil = 12;
    int ncol = 12;
    int N = nfil * ncol;

    cudaEvent_t start, stop;
    float time;

    size_t size = N * sizeof(float);

    A_h = (float *)malloc(size);   
    B_h = (float *)malloc(size);
    C_h = (float *)malloc(size);

    

    for (int i = 0; i < nfil; i++) {
        for (int j = 0; j < ncol; j++) {
            A_h[i*ncol+j] = rand() % 10;
            B_h[i*ncol+j] = rand() % 10;
        }
    }

    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    dim3 block_size(32, 32);
    dim3 numBlocks(1, 1);

    addKernel << <  numBlocks, block_size  >> > (C_d, A_d, B_d, nfil, ncol);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    printf("\n \nMatriz c: \n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%d", C_h[i * ncol + j]);
        }
        printf("\n");
    }

    free(A_h);
    free(B_h);
    free(C_h);

    cudaFree(C_d);
    cudaFree(A_d);
    cudaFree(B_d);

    return 0;
}

