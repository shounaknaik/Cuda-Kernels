// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
#include <string>
// #include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


//declaring special functions to be used in this code.
extern "C" void MatVecMul(const float* A, const float* x, float* y, int M, int N);
extern "C" void SigmoidActivation(float* x, int N);
extern "C" void ComputeGradient(const float* inputs, const float* errors, float* gradients, int inputSize);

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Only Binary Classification
void ReadCIFAR10BinaryClassification(const std::string& filename, float* images, float* labels, int& numLoaded, int maxImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }

    const int imageSize = 3072; // 32x32x3
    unsigned char buffer[imageSize + 1]; // +1 for the label

    numLoaded = 0;
    while (numLoaded < maxImages && file.read(reinterpret_cast<char*>(buffer), imageSize + 1)) {
        unsigned char label = buffer[0];

        // Only select classes 0 and 1
        if (label == 0 || label == 1) {
            labels[numLoaded] = static_cast<float>(label);

            // Copy and normalize image data
            for (int i = 0; i < imageSize; ++i) {
                images[numLoaded * imageSize + i] = static_cast<float>(buffer[i + 1]) / 255.0f;
            }
            numLoaded++;
        }
    }

    file.close();
}

void Forward(const float* d_weights, const float* d_input, float* d_output, int inputSize) {
    // Launch matrix-vector multiplication kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (1 + threadsPerBlock - 1) / threadsPerBlock; // Output size is 1
    //If dimension of output greater, different blocks process different modalities.
    //Here 256 examples will be processed in that batch.

    MatVecMul << <blocksPerGrid, threadsPerBlock >> > (d_weights, d_input, d_output, 1, inputSize);
    cudaCheckError(cudaPeekAtLastError());

    // Apply sigmoid activation
    SigmoidActivation <<<blocksPerGrid, threadsPerBlock >>> (d_output, 1);
    cudaCheckError(cudaPeekAtLastError());
}


// Training Function
void Train(float* h_inputs, float* h_labels, float* h_weights, int inputSize, int numSamples, int epochs, float learningRate) {
    // Allocate device memory for weights
    float* d_weights;
    cudaMalloc(&d_weights, inputSize * sizeof(float));
    cudaMemcpy(d_weights, h_weights, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;

        for (int sample = 0; sample < numSamples; ++sample) {
            // Get the current sample
            float* h_inputSample = &h_inputs[sample * inputSize];
            float h_label = h_labels[sample];

            // Allocate device memory for input and output
            float* d_inputSample;
            float* d_output;
            cudaMalloc(&d_inputSample, inputSize * sizeof(float));
            cudaMalloc(&d_output, sizeof(float));

            cudaMemcpy(d_inputSample, h_inputSample, inputSize * sizeof(float), cudaMemcpyHostToDevice);

            // Forward pass
            Forward(d_weights, d_inputSample, d_output, inputSize);

            // Copy prediction back to host
            float h_prediction;
            cudaMemcpy(&h_prediction, d_output, sizeof(float), cudaMemcpyDeviceToHost);

            // Compute error
            float h_error = h_prediction - h_label;
            totalLoss += 0.5f * h_error * h_error; // Mean Squared Error

            // Allocate device memory for error and gradient
            float* d_error;
            float* d_gradient;
            cudaMalloc(&d_error, sizeof(float));
            cudaMalloc(&d_gradient, inputSize * sizeof(float));

            cudaMemcpy(d_error, &h_error, sizeof(float), cudaMemcpyHostToDevice);

            // Compute gradient
            int threadsPerBlock = 256;
            int blocksPerGrid = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
            ComputeGradient << <blocksPerGrid, threadsPerBlock >> > (d_inputSample, d_error, d_gradient, inputSize);
            cudaCheckError(cudaPeekAtLastError());

            // Update weights
            // Copy gradient back to host
            float* h_gradient = new float[inputSize];
            cudaMemcpy(h_gradient, d_gradient, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < inputSize; ++i) {
                h_weights[i] -= learningRate * h_gradient[i];
            }

            // Copy updated weights back to device
            cudaMemcpy(d_weights, h_weights, inputSize * sizeof(float), cudaMemcpyHostToDevice);

            // Free allocated memory
            delete[] h_gradient;
            cudaFree(d_inputSample);
            cudaFree(d_output);
            cudaFree(d_error);
            cudaFree(d_gradient);
        }

        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / numSamples << std::endl;
    }

    // Free device memory
    cudaFree(d_weights);
}


// Evaluation Function
void Evaluate(float* h_inputs, float* h_labels, float* h_weights, int inputSize, int numSamples) {
    int correct = 0;

    // Allocate device memory for weights
    float* d_weights;
    cudaMalloc(&d_weights, inputSize * sizeof(float));
    cudaMemcpy(d_weights, h_weights, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    for (int sample = 0; sample < numSamples; ++sample) {
        float* h_inputSample = &h_inputs[sample * inputSize];
        float h_label = h_labels[sample];

        // Allocate device memory for input and output
        float* d_inputSample;
        float* d_output;
        cudaMalloc(&d_inputSample, inputSize * sizeof(float));
        cudaMalloc(&d_output, sizeof(float));

        cudaMemcpy(d_inputSample, h_inputSample, inputSize * sizeof(float), cudaMemcpyHostToDevice);

        // Forward pass
        Forward(d_weights, d_inputSample, d_output, inputSize);

        // Copy prediction back to host
        float h_prediction;
        cudaMemcpy(&h_prediction, d_output, sizeof(float), cudaMemcpyDeviceToHost);

        // Apply threshold to determine predicted class
        int predictedLabel = h_prediction >= 0.5f ? 1 : 0;

        if (predictedLabel == static_cast<int>(h_label)) {
            correct++;
        }

        // Free allocated memory
        cudaFree(d_inputSample);
        cudaFree(d_output);
    }

    float accuracy = static_cast<float>(correct) / numSamples * 100.0f;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    cudaFree(d_weights);
}



// Main Function
int main() {
    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Define network parameters
    const int inputSize = 32 * 32 * 3; // CIFAR-10 images
    const int maxSamples = 10000; // Maximum samples to load
    const int epochs = 20;
    const float learningRate = 0.01f;

    // Allocate host memory for inputs, labels, and weights
    float* h_inputs = new float[maxSamples * inputSize];
    float* h_labels = new float[maxSamples];
    float* h_weights = new float[inputSize];

    // Initialize weights with small random values
    for (int i = 0; i < inputSize; ++i) {
        h_weights[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.01f;
    }

    // Load CIFAR-10 training data (adjust the path to your data file)
    int numTrainSamples = 0;
    std::string trainBatchFile = "../data/cifar-10-batches-bin/data_batch_1.bin"; // Replace with your path
    ReadCIFAR10BinaryClassification(trainBatchFile, h_inputs, h_labels, numTrainSamples, maxSamples);

    if (numTrainSamples == 0) {
        std::cerr << "No training data loaded. Exiting." << std::endl;
        return -1;
    }

    std::cout << "Number of training samples loaded: " << numTrainSamples << std::endl;

    // Train the network
    Train(h_inputs, h_labels, h_weights, inputSize, numTrainSamples, epochs, learningRate);

    // Load CIFAR-10 test data (adjust the path to your test data file)
    float* h_testInputs = new float[maxSamples * inputSize];
    float* h_testLabels = new float[maxSamples];
    int numTestSamples = 0;
    std::string testBatchFile = "../data/cifar-10-binary/cifar-10-batches-bin/test_batch.bin"; // Replace with your path
    ReadCIFAR10BinaryClassification(testBatchFile, h_testInputs, h_testLabels, numTestSamples, maxSamples);

    if (numTestSamples == 0) {
        std::cerr << "No test data loaded. Exiting." << std::endl;
        return -1;
    }

    std::cout << "Number of test samples loaded: " << numTestSamples << std::endl;

    // Evaluate the network
    Evaluate(h_testInputs, h_testLabels, h_weights, inputSize, numTestSamples);

    // Free host memory
    delete[] h_inputs;
    delete[] h_labels;
    delete[] h_weights;
    delete[] h_testInputs;
    delete[] h_testLabels;

    return 0;
}


