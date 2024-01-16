//
// Created by mgjer on 15/01/2024.
//

#include <iostream>
#include <random>
#include <rasterize_points.h>

#include "Viewer/Cuda/Cuda_example.h"
#include "Viewer/Tools/Logger.h"

#define CHECK_CUDA_ERROR(val) check_cuda_result((val), (#val), __FILE__, __LINE__)

template<typename T>
void check_cuda_result(T result, const char *const func, const char *const file, int line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " with code=" << ((unsigned int) result) << ", " << cudaGetErrorString(result) << ", \"" << func << "\"?" << std::endl;
    }

}

__global__ void sum_and_filter(int *a, int *b, int *c, int *d, unsigned int n)
{
    unsigned int i = threadIdx.y * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }

    while (i < n)
    {
        c[i] = a[i] + b[i];
        i += gridDim.x * gridDim.y;
    }

    __syncthreads();
    // Now all data in c has been evaluated, and therefore we can actually READ data from c!

    i = threadIdx.y * blockDim.x + threadIdx.x;
    while (i < n)
    {
        int prev = (i - 1) < 0 ? 0 : c[i - 1];
        int next = (i + 1) >= n ? 0 : c[i + 1];
        d[i] = (c[i] + prev + next) / 3;
        i += gridDim.x * gridDim.y;
    }
}


__global__ void sum(int *a, int *b, int *c, unsigned int n)
{
    unsigned int i = blockIdx.y * gridDim.x + blockIdx.x;
    c[i] = a[i] + b[i];
}

int function_with_cuda_calls(){
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    constexpr unsigned int N = 100;
    std::uniform_int_distribution<int> distrib(1, 10);
    std::random_device dev;

    int a[N], b[N], c[N] = { 0 };
    for (unsigned int i = 0; i < N; i++)
    {
        a[i] = distrib(dev);
        b[i] = distrib(dev);
    }

    int *a_cuda, *b_cuda, *c_cuda;
    CHECK_CUDA_ERROR(cudaMalloc(&a_cuda, sizeof(a)));
    CHECK_CUDA_ERROR(cudaMalloc(&b_cuda, sizeof(b)));
    CHECK_CUDA_ERROR(cudaMalloc(&c_cuda, sizeof(c)));
    CHECK_CUDA_ERROR(cudaMemcpy(a_cuda, a, sizeof(a), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(b_cuda, b, sizeof(b), cudaMemcpyHostToDevice));

    int num_blocks_x = static_cast<int>(sqrt(N));
    int num_blocks_y = (N + (num_blocks_x - 1)) / num_blocks_x;
    dim3 num_blocks(num_blocks_x, num_blocks_y, 1);

    sum<<<num_blocks, 1>>>(a_cuda, b_cuda, c_cuda, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(c, c_cuda, sizeof(c), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N / 4; i++){
        Log::Logger::getInstance()->info("{} + {} = {}", a[i], b[i], c[i]);
    }

    CHECK_CUDA_ERROR(cudaFree(a_cuda));
    CHECK_CUDA_ERROR(cudaFree(b_cuda));
    CHECK_CUDA_ERROR(cudaFree(c_cuda));
    return 0;
}

int call_rasterize_functions() {
    // Creating dummy tensors
    torch::Tensor background = torch::rand({3, 3});  // Adjust the size as per your requirements
    torch::Tensor means3D = torch::rand({3, 3});
    torch::Tensor colors = torch::rand({3, 3});
    torch::Tensor opacity = torch::rand({3, 3});
    torch::Tensor scales = torch::rand({3, 3});
    torch::Tensor rotations = torch::rand({3, 3});
    torch::Tensor cov3D_precomp = torch::rand({3, 3});
    torch::Tensor viewmatrix = torch::rand({3, 3});
    torch::Tensor projmatrix = torch::rand({3, 3});
    torch::Tensor sh = torch::rand({3, 3});
    torch::Tensor campos = torch::rand({3});

    // Other parameters
    float scale_modifier = 1.0f;
    float tan_fovx = 0.5f;
    float tan_fovy = 0.5f;
    int image_height = 480;
    int image_width = 640;
    int degree = 2;
    bool prefiltered = false;
    bool debug = true;

    // Call the function
    auto result = RasterizeGaussiansCUDA(
            background, means3D, colors, opacity, scales, rotations,
            scale_modifier, cov3D_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy,
            image_height, image_width, sh, degree, campos, prefiltered, debug
    );

    Log::Logger::getInstance()->info("Called cuda diff rasterizer");
    return 0;
}