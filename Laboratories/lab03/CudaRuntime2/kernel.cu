#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Kernel mult
__global__ void matrixMultiply(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Kernel sum
__global__ void matrixAdd(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Config matriz
    const int width = 100;
    const int height = 500;
    const int size = width * height;

    // Alojamiento y asignación de mem HOST
    float* h_A = new float[size];
    float* h_B = new float[size];
    float* h_C = new float[size];

    // Inicializar matrices
    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // Alojamiento de memoria
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size * sizeof(float));
    cudaMalloc((void**)&d_B, size * sizeof(float));
    cudaMalloc((void**)&d_C, size * sizeof(float));

    // HOST -> DEVICE
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    // Config cuadrícula
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (width + blockDim.y - 1) / blockDim.y);

    // Medición del tiempo
    auto start = std::chrono::high_resolution_clock::now();

    // Lanzamiento kernel
    matrixMultiply << <gridDim, blockDim >> > (d_A, d_B, d_C, width);
   /* matrixAdd << <gridDim, blockDim >> > (d_A, d_B, d_C, width);*/

    // Error Management
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "Error in multiplication: " << cudaGetErrorString(cudaError) << std::endl;
        return -1;
    }

    // DEVICE -> HOST
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Tiempo de ejecución de la multiplicación: " << duration.count() << " secs" << std::endl;

    // Liberar mem DEVICE
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Liberar mem HOST
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
