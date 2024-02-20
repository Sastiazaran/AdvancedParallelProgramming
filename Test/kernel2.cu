#include <cuda_runtime.h>

__global__ void segmentationKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        outputImage[index] = (inputImage[index] > threshold) ? 1 : 0;
    }
}

int main() {
    
    int width = 640;
    int height = 480;

    
    int threshold = 154;

    
    size_t imageSize = width * height * sizeof(unsigned char);

    
    unsigned char* h_inputImage = (unsigned char*)malloc(imageSize);

    
    for (int i = 0; i < width * height; ++i) {
        h_inputImage[i] = rand() % 256;  
    }

    
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    cudaMalloc((void**)&d_inputImage, imageSize);
    cudaMalloc((void**)&d_outputImage, imageSize);

    
    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    
    segmentationKernel << <gridSize, blockSize >> > (d_inputImage, d_outputImage, width, height, threshold);

    
    cudaMemcpy(h_inputImage, d_inputImage, imageSize, cudaMemcpyDeviceToHost);

    
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

     
    free(h_inputImage);

    return 0;
}
