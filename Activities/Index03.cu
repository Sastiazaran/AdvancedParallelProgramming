
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>

using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__)}

cudaError_t addWithCuda();

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: $s $s $d\n\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void queryDevice() {
    int d_Count = 0;

    cudaGetDeviceCount(&d_Count);

    if (d_Count == 0) {
        printf("No CUDA devide found: \n\r");
    }

    cudaDeviceProp(prop);

    for (int devNo = 0; devNo < d_Count; devNo++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, devNo);
        printf("    Device Number:                      %d\n", devNo);
        printf("    Device Name:                        %d\n", prop.name);
        printf("    No. of multiprocessors:             %d\n", prop.multiProcessorCount);
        printf("    Compute capability:                 %d\n", prop.major, prop.minor);
        printf("    Memory clock rate (Khz):            %d\n", prop.memoryClockRate );
        printf("    Memory bus rate (bits):             %d\n", prop.memoryBusWidth);
        printf("    Peak Memory Bandwidth(GB/s):        %8.2f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("    Total amount of Global Memory:      %dKB\n", prop.totalGlobalMem / 1024);
        printf("    Total amount of const memory:       %dKB\n", prop.totalConstMem / 1024 );
        printf("    total of shared memory per block:   %dKB\n", prop.sharedMemPerBlock / 1024 );
        printf("    Total of shared memory per mp:      %dKB\n", prop.sharedMemPerMultiprocessor / 1024 );
        printf("    Warp Size:                          %d\n", prop.warpSize);
        printf("    Max. threads per block:             %d\n", prop.maxThreadsPerBlock );
        printf("    Max. threads per MP:                %d\n", prop.maxThreadsPerMultiProcessor);
        printf("    Max. number of warps per MP:        %d\n", prop.maxThreadsPerMultiProcessor / 32);
        printf("    Max. Grid Size:                     (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("    Max. block dimension:               (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }

}

int main() {
    int A_rows = 400;
    int A_cols = 640;
    int A_size_bytes = A_rows * A_cols * sizeof(int);
    int* h_a = (int*)malloc(A_size_bytes);
    int* d_a, * d_b, * d_c, * d_out;

    GPUErrorAssertion(cudaMalloc((int**)&d_a, A_size_bytes));

    queryDevice();
    return 0;
}

//__global__ void print_all_idx()
//{
//    int tidx = threadIdx.x;
//    int tidy = threadIdx.y;
//    int tidz = threadIdx.z;
//
//    int bidx = blockIdx.x;
//    int bidy = blockIdx.y;
//    int bidz = blockIdx.z;
//
//    int gdimx = gridDim.x;
//    int gdimy = gridDim.y;
//    int gdimz = gridDim.z;
//
//    printf("[DEVICE] threadId.x : %d, blockId.x : %d, gridId.x : %d \n", tidx, bidx, gdimx);
//    printf("[DEVICE] threadId.y : %d, blockId.y : %d, gridId.y : %d \n", tidy, bidy, gdimy);
//    printf("[DEVICE] threadId.z : %d, blockId.z : %d, gridId.z : %d \n", tidz, bidz, gdimz);
//}

//int main()
//{
//    //initialization
//    dim3 blockSize(4, 4, 4);
//    dim3 gridSize(2, 2, 2);
//    int* c_cpu;
//    int* a_cpu;
//    int* b_cpu;
//    int* c_device;
//    int* a_device;
//    int* b_device;
//    const int data_count = 10000;
//    const int data_size = data_count * sizeof(int);
//
//    c_cpu = (int*)malloc(data_size);
//    a_cpu = (int*)malloc(data_size);
//    b_cpu = (int*)malloc(data_size);
//
//    //Memory allocation
//    cudaMalloc((void**)&c_device, data_size);
//    cudaMalloc((void**)&a_device, data_size);
//    cudaMalloc((void**)&b_device, data_size);
//
//    //transfer to CPU host to GPU device
//    cudaMemcpy(c_device, c_cpu, data_size, cudaMemcpyHostToDevice);
//    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
//    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);
//
//    //launch kernel
//    print_all_idx << < gridSize, blockSize >> > ();
//
//    //transfer to GPU host to CPU device
//    cudaMemcpy(c_cpu, c_device, data_size, cudaMemcpyDeviceToHost);
//    cudaMemcpy(a_cpu, a_device, data_size, cudaMemcpyDeviceToHost);
//    cudaMemcpy(b_cpu, b_device, data_size, cudaMemcpyDeviceToHost);
//
//    cudaDeviceReset();
//    cudaFree(c_device);
//    cudaFree(a_device);
//    cudaFree(b_device);
//
//    //inicializar datos
//    //inicializar grid
//    //transferir CPU A GPU
//    // Lanzar Kernel
//    //transferir de Gpu a CPU
//
//
//    return 0;
//}

