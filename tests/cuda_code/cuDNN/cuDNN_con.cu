#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

#define BATCH_SIZE 1
#define CHANNELS 1
#define HEIGHT 5
#define WIDTH 5
#define KERNEL_SIZE 3

int main() {
    // cuDNN 핸들 생성
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // 입력 텐서 설정
    cudnnTensorDescriptor_t inputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, BATCH_SIZE, CHANNELS, HEIGHT, WIDTH);

    // 커널(필터) 설정
    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, CHANNELS, CHANNELS, KERNEL_SIZE, KERNEL_SIZE);

    // 출력 텐서 크기 계산
    int out_n, out_c, out_h, out_w;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &out_n, &out_c, &out_h, &out_w);

    // 출력 텐서 설정
    cudnnTensorDescriptor_t outputDesc;
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);

    // 메모리 할당
    float input[HEIGHT * WIDTH] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float kernel[KERNEL_SIZE * KERNEL_SIZE] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, HEIGHT * WIDTH * sizeof(float));
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_output, out_h * out_w * sizeof(float));
    cudaMemcpy(d_input, input, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // 합성곱 연산 실행
    const float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle, &alpha, inputDesc, d_input, filterDesc, d_kernel, convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, outputDesc, d_output);

    // 결과를 호스트로 복사
    float output[out_h * out_w];
    cudaMemcpy(output, d_output, out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    // 출력 결과
    std::cout << "Output:\n";
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            std::cout << output[i * out_w + j] << " ";
        }
        std::cout << std::endl;
    }

    // 리소스 해제
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroy(handle);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
