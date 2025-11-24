// micro_gemm_4x4.cu
#include <cstdio>

__global__ void micro_gemm_4x4(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int K)
{
    // C 4x4 block in registers
    float c00=0, c01=0, c02=0, c03=0;
    float c10=0, c11=0, c12=0, c13=0;
    float c20=0, c21=0, c22=0, c23=0;
    float c30=0, c31=0, c32=0, c33=0;

    for (int k = 0; k < K; k++) {
        // load A(4) and B(4) into registers
        float a0 = A[0*K + k];
        float a1 = A[1*K + k];
        float a2 = A[2*K + k];
        float a3 = A[3*K + k];

        float b0 = B[k*4 + 0];
        float b1 = B[k*4 + 1];
        float b2 = B[k*4 + 2];
        float b3 = B[k*4 + 3];

        // Micro-kernel FMAs (4×4)
        c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
        c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
        c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
        c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
    }

    // single thread writes C
    C[0] = c00; C[1] = c01; C[2] = c02; C[3] = c03;
    C[4] = c10; C[5] = c11; C[6] = c12; C[7] = c13;
    C[8] = c20; C[9] = c21; C[10]= c22; C[11]= c23;
    C[12]= c30; C[13]= c31; C[14]= c32; C[15]= c33;
}

int main() {
    int K = 1024;

    float *A, *B, *C;
    cudaMallocManaged(&A, sizeof(float)*4*K);
    cudaMallocManaged(&B, sizeof(float)*K*4);
    cudaMallocManaged(&C, sizeof(float)*16);

    // init
    for (int i=0;i<4*K;i++) A[i] = (i%7)*0.1f;
    for (int i=0;i<K*4;i++) B[i] = (i%5)*0.2f;

    micro_gemm_4x4<<<1,1>>>(A,B,C,K);
    cudaDeviceSynchronize();

    printf("C[0..3]: %f %f %f %f\n", C[0],C[1],C[2],C[3]);

    cudaFree(A); cudaFree(B); cudaFree(C);
}
