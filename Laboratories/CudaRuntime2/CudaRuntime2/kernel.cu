
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void vectorBitwiseOR(const unsigned char* A, const unsigned char* B, unsigned char* C, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] | B[tid];
    }
}

int main()
{

    const int size_in_bytes = 1024;
    unsigned char A[size_in_bytes];
    unsigned char B[size_in_bytes];
    unsigned char C[size_in_bytes];
    int N_size;
    int total_threads;
    int threads_per_block;

    int blocks;

    N_size = size_in_bytes / sizeof(unsigned char);
    total_threads = 1024 * 46* 16;
    threads_per_block = total_threads / 16;
    blocks = (N_size + threads_per_block - 1) / threads_per_block;

    vectorBitwiseOR << <blocks, threads_per_block >> > (A, B, C, N_size);

    printf("Nsize: %d\n", total_threads);
    printf("Threads per block: %d", threads_per_block);

    return 0;

}
