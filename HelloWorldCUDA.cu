#include <iostream>
#include <cuda_runtime.h>

using namespace std;
__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}

int main() {
    // Launch the kernel with 1 block and 1 thread
    helloFromGPU<<<1, 1>>>();

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    std::cout << "Hello from CPU!" << endl;
    return 0;
}
