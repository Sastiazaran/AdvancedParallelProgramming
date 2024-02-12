
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

cudaError_t addWithCuda();

__global__ void print_all_idx()
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bidz = blockIdx.z;

    int gdimx = gridDim.x;
    int gdimy = gridDim.y;
    int gdimz = gridDim.z;

    printf("[DEVICE] threadId.x : %d, blockId.x : %d, gridId.x : %d \n", tidx, bidx, gdimx);
    printf("[DEVICE] threadId.y : %d, blockId.y : %d, gridId.y : %d \n", tidy, bidy, gdimy);
    printf("[DEVICE] threadId.z : %d, blockId.z : %d, gridId.z : %d \n", tidz, bidz, gdimz);
}

int main()
{
    //initialization
    dim3 blockSize(4,4,4);
    dim3 gridSize(2, 2, 2);
    int* c_cpu;
    int* a_cpu;
    int* b_cpu;
    int* c_device;
    int* a_device;
    int* b_device;
    const int data_count = 10000;
    const int data_size = data_count * sizeof(int);

    c_cpu = (int*)malloc(data_size);
    a_cpu = (int*)malloc(data_size);
    b_cpu = (int*)malloc(data_size);

    //Memory allocation
    cudaMalloc((void**)&c_device, data_size);
    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);

    //transfer to CPU host to GPU device
    cudaMemcpy(c_device, c_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);

    //launch kernel
    print_all_idx << < gridSize, blockSize >> >();

    //transfer to GPU host to CPU device
    cudaMemcpy(c_cpu, c_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_cpu, a_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_device, data_size, cudaMemcpyDeviceToHost);

    cudaDeviceReset();
    cudaFree(c_device);
    cudaFree(a_device);
    cudaFree(b_device);

    //inicializar datos
    //inicializar grid
    //transferir CPU A GPU
    // Lanzar Kernel
    //transferir de Gpu a CPU


    return 0;
}

