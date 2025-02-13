#include <iostream>
#include <cuda_runtime.h>
#include <cuda/std/atomic>


#define N 16

__global__ void matrixMul(float *A, float *B, float *C, int n){

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < n && col < n){
        float sum = 0;
        for (int i = 0; i < n ;++i){
            sum = sum + A[row*n+i] + B[i*n+col];
        }
        C[row*n+col] = sum;
 
    }

}

int main(){
int size = N*N*sizeof(float);

float *h_A = (float *)malloc(size);
float *h_B = (float *)malloc(size);
float *h_C = (float *)malloc(size);

for (int i = 0; i < N*N; i++){
    h_A[i] =  1.2*i;
    h_B[i] =  1.4*i;
}

float *d_A, *d_B , *d_C;
cudaMalloc((void **)&d_A, size);
cudaMalloc((void **)&d_B, size);
cudaMalloc((void **)&d_C, size);

cudaMemcpy(d_A,h_A,size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B,h_B,size, cudaMemcpyHostToDevice);

dim3 threadsPerBlock(16,16);
dim3 blocksPerGrid((N+15)/16, (N+15)/16);

matrixMul <<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,N);
cudaMemcpy(h_C,d_C, size , cudaMemcpyDeviceToHost);

printf("resulting matrix:\n");

for(int i = 0;i<16;i++){
    for (int j =0;j<16;j++){
            printf("%f  ", h_C[i*N+j]);
    }

    printf("\n");

}

free(h_A);
free(h_B);
free(h_C);

cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

return 0 ;


}