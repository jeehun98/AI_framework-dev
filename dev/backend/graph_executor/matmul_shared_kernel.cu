#define TILE_WIDTH 16

__global__ void matmul_shared_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                     int A_rows, int A_cols, int B_cols) {
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;  // output row
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;  // output column

    float sum = 0.0f;

    for (int ph = 0; ph < (A_cols + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        // 각 타일 블록 로딩: row-major 방식
        if (row < A_rows && ph * TILE_WIDTH + threadIdx.x < A_cols)
            A_tile[threadIdx.y][threadIdx.x] = A[row * A_cols + ph * TILE_WIDTH + threadIdx.x];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (ph * TILE_WIDTH + threadIdx.y < A_cols && col < B_cols)
            B_tile[threadIdx.y][threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * B_cols + col];
        else
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // 타일 로딩 완료 대기

        // 누적 곱
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];

        __syncthreads();  // 다음 타일 준비
    }

    // 결과 저장
    if (row < A_rows && col < B_cols)
        C[row * B_cols + col] = sum;
}
