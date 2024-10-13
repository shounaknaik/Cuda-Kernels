#include <stdio.h>

void helloCpu()
{
    printf("Hello from the CPU \n");
}

__global__ void helloGPU(){
    printf("Hello from GPU");
}

int main(){
    helloGPU<<<1,1>>>(); 
    cudaDeviceSynchronize();
    return 0;

}

