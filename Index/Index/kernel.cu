
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void print()
{
    int i = threadIdx.x;
    printf("[DEVICE] ThreadIdx.x: %d\n", i);
}

int main()
{
    print << <2, 8 >> > ();

    return 0;
}


