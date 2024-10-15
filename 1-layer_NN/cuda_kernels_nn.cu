//Specialized kernels for 1 layer Neural Network
#include <cuda_runtime.h>

extern "C"{
    //ensures we can call the cuda kernel from a cpp code



__global__ void MatVecMul(const float* A, const float* x, float* y, int M, int N) {
    // Each thread computes one element of y- one prediction for the example
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            sum += A[row * N + col] * x[col];
        }
        y[row] = sum;
    }
}


__global__ void SigmoidActivation(float* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        x[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void ComputeGradient(const float* inputs, const float* errors, float* gradients, int inputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < inputSize) {
        gradients[idx] = inputs[idx] * errors[0]; // Since output size is 1 for binary classification
    }
}

__global__ void ComputeOutputError(const float* predictions, const float* labels, float* errors, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        errors[idx] = predictions[idx] - labels[idx];
    }
}

}

