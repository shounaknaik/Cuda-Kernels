// Program to computer matrix multiplication using CUDA

#include <cstdlib>
#include <iostream>
#include <chrono>
// using namespace std

__global__ void matrixMul(int *a,int *b, int *c, int N){
    //Calculate global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //Boundary check
    if (row < N  && col < N){
        int temp = 0;
        for (int i = 0;i<N;i++){
            temp += a[row*N + i] * b[i*N + col];

        }

        //Write back result
        c[row*N + col] = temp;
    }

}

//CPU Function
void multiply(int *a,int *b,int *c, int N){

    int temp;
    for (int i = 0;i<N;i++){
        for (int j = 0; j<N;j++){

            temp = 0;
            for(int k =0;k<N;k++){
                temp += a[i*N +k]*b[k*N +j];
            }
            c[i*N + j] = temp;

        }
        
        
    }

}

//initialize a square matrix with random numbers between 0-100
void init_matrix(int *m,int N){
    for (int i = 0;i<N*N;i++){
        m[i] = rand()%100;
    }

}

int main(){

    int N = 1 << 10; //2^10

    //allocate memory for mnatrices
    size_t bytes = N*N*sizeof(int);
    int *a,*b,*c,*d;
    cudaMallocManaged(&a,bytes);
    cudaMallocManaged(&b,bytes);
    cudaMallocManaged(&c,bytes);
    cudaMallocManaged(&d,bytes);

    

    //Initialize data
    init_matrix(a,N);
    init_matrix(b,N);

    //set threadsperBlock and blocksperGrids 
    int threads = 16;
    int blocks = (N+threads -1)/threads;

    // set up our kernel launch parameters
    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks,blocks);

    //Maintain timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timer
    cudaEventRecord(start);

    // Launch our kernel
    matrixMul<<<BLOCKS,THREADS>>>(a,b,c,N);
    //Record end time
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms,start,stop);
    std::cout<<"GPU execution in "<<ms<<" milliseconds"<<std::endl;

    //Measure CPU Function
    // delete start;
    auto st = std::chrono::high_resolution_clock::now();
    multiply(a,b,d,N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - st;

    std::cout<<"CPU execution in "<<elapsed.count()<<" seconds"<<std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);

    return 0;


}