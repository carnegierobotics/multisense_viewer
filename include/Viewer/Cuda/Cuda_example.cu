//
// Created by mgjer on 15/01/2024.
//

#include <iostream>
#include <random>
#include <rasterize_points.h>

#include "Viewer/Cuda/Cuda_example.h"
#include "Viewer/Tools/Logger.h"
#include <opencv2/opencv.hpp>

#define CHECK_CUDA_ERROR(val) check_cuda_result((val), (#val), __FILE__, __LINE__)

template<typename T>
void check_cuda_result(T result, const char *const func, const char *const file, int line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " with code=" << ((unsigned int) result) << ", "
                  << cudaGetErrorString(result) << ", \"" << func << "\"?" << std::endl;
    }

}

__global__ void sum_and_filter(int *a, int *b, int *c, int *d, unsigned int n) {
    unsigned int i = threadIdx.y * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    while (i < n) {
        c[i] = a[i] + b[i];
        i += gridDim.x * gridDim.y;
    }

    __syncthreads();
    // Now all data in c has been evaluated, and therefore we can actually READ data from c!

    i = threadIdx.y * blockDim.x + threadIdx.x;
    while (i < n) {
        int prev = (i - 1) < 0 ? 0 : c[i - 1];
        int next = (i + 1) >= n ? 0 : c[i + 1];
        d[i] = (c[i] + prev + next) / 3;
        i += gridDim.x * gridDim.y;
    }
}


__global__ void sum(int *a, int *b, int *c, unsigned int n) {
    unsigned int i = blockIdx.y * gridDim.x + blockIdx.x;
    c[i] = a[i] + b[i];
}

int function_with_cuda_calls() {
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    constexpr unsigned int N = 100;
    std::uniform_int_distribution<int> distrib(1, 10);
    std::random_device dev;

    int a[N], b[N], c[N] = {0};
    for (unsigned int i = 0; i < N; i++) {
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

    for (int i = 0; i < N / 4; i++) {
        Log::Logger::getInstance()->info("{} + {} = {}", a[i], b[i], c[i]);
    }

    CHECK_CUDA_ERROR(cudaFree(a_cuda));
    CHECK_CUDA_ERROR(cudaFree(b_cuda));
    CHECK_CUDA_ERROR(cudaFree(c_cuda));
    return 0;
}

#include <torch/torch.h>
#include <vector>


// GaussianData class definition
class GaussianData {
public:
    torch::Tensor xyz;
    torch::Tensor rot;
    torch::Tensor scale;
    torch::Tensor opacity;
    torch::Tensor sh;

    torch::Tensor flat() const {
        return torch::cat({xyz, rot, scale, opacity, sh}, -1).contiguous();
    }

    size_t length() const {
        return xyz.size(0);
    }

    int sh_dim() const {
        return sh.size(-1);
    }
};

torch::Tensor ConvertGlmMat4ToTensor(const glm::mat4 &mat) {
    // Create a 4x4 tensor
    torch::Tensor tensor = torch::empty({4, 4}, torch::kFloat32);

    // Copy data from glm::mat4 to the tensor
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            tensor[i][j] = mat[i][j];
        }
    }
    return tensor;
}

torch::Tensor ConvertGlmVec3ToTensor(const glm::vec3 &vec) {
    // Create a 1D tensor with 3 elements
    torch::Tensor tensor = torch::empty({3}, torch::kFloat32);

    // Copy data from glm::vec3 to the tensor
    tensor[0] = vec.x;
    tensor[1] = vec.y;
    tensor[2] = vec.z;

    return tensor;
}

GaussianData naive_gaussian() {
    auto gau_xyz = torch::tensor({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}, torch::dtype(torch::kFloat32)).view({-1, 3});
    auto gau_rot = torch::tensor({1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, torch::dtype(torch::kFloat32)).view(
            {-1, 4});
    auto gau_s = torch::tensor({0.03, 0.03, 0.03, 0.2, 0.03, 0.03, 0.03, 0.2, 0.03, 0.03, 0.03, 0.2},
                               torch::dtype(torch::kFloat32)).view({-1, 3});
    auto gau_c = torch::tensor({1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1}, torch::dtype(torch::kFloat32)).view({-1, 3});
    gau_c = (gau_c - 0.5) / 0.28209;
    auto gau_a = torch::tensor({1, 1, 1, 1}, torch::dtype(torch::kFloat32)).view({-1, 1});

    return GaussianData(gau_xyz, gau_rot, gau_s, gau_a, gau_c);
}

CudaImplementation::CudaImplementation(const CudaImplementation::RasterSettings *settings) {
    torch::Device device(torch::kCUDA);

    viewmatrix =  ConvertGlmMat4ToTensor(settings->viewMat).to(device);
    projmatrix =  ConvertGlmMat4ToTensor(settings->projMat).to(device);
    campos     =  ConvertGlmVec3ToTensor(settings->camPos).to(device);
    bg         =  torch::tensor({0.0, 0.0, 0.0}, torch::dtype(torch::kFloat32)).to(device);
    // Other parameters
    scale_modifier  = settings->scaleModifier;
    tan_fovx        = settings->tanFovX;
    tan_fovy        = settings->tanFovY;
    image_height    = settings->imageHeight;
    image_width     = settings->imageWidth;
    degree          = settings->shDegree;
    prefiltered     = settings->prefilter;
    debug           = settings->debug;

    auto gaussianData = naive_gaussian();
    // Example usage
    auto flatData = gaussianData.flat();

    means3D         = gaussianData.xyz.to(device);
    shs             = gaussianData.sh.to(device);
    opacity         = gaussianData.opacity.to(device);
    scales          = gaussianData.scale.to(device);
    rotations       = gaussianData.rot.to(device);
    cov3D_precomp   = torch::tensor({}).to(device);
    colors          = torch::tensor({}).to(device);

}

void CudaImplementation::draw() {

    int rendered;
    torch::Tensor out_color, radii, geomBuffer, binningBuffer, imgBuffer;
    // Call the function
    std::tie(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer) = RasterizeGaussiansCUDA(
            bg, means3D, colors, opacity, scales, rotations,
            scale_modifier, cov3D_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy,
            image_height, image_width, shs, degree, campos, prefiltered, debug
    );

    Log::Logger::getInstance()->info("Called cuda diff rasterizer. Num rendered: {}", rendered);
    /*
    // Ensure the tensor is on CPU and is of the correct type (e.g., byte tensor)
    if (out_color.device().is_cuda()) {
        out_color = out_color.to(torch::kCPU);
    }
    out_color = out_color.to(torch::kU8);  // Assuming out_color is a float tensor

    // Convert to a 3-channel (RGB) image if necessary
    if (out_color.sizes().size() == 2) {  // If it's a single channel (grayscale) image
        out_color = out_color.unsqueeze(2).repeat({1, 1, 3});  // Convert to RGB
    }

    // Convert torch::Tensor to cv::Mat
    cv::Mat image(out_color.size(0), out_color.size(1), CV_8UC3, out_color.data_ptr<uchar>());

    // Display the image
    cv::imshow("Output Image", image);
    cv::waitKey(1); // Wait for a key press (use 0 for infinite wait)

     */
    auto img = out_color.permute({1, 2, 0});  // Change [Channels, Height, Width] to [Height, Width, Channels]

    // Add an alpha channel
    //auto alpha_channel = torch::ones_like(img.index({"...", Slice(nullptr, 1)}));
    //img = torch::cat({img, alpha_channel}, -1);

    img = img.contiguous();

        // Extract height and width
    int64_t height = img.size(0);
    int64_t width = img.size(1);

    // Ensure the tensor is on the CPU and is a byte tensor
    if (img.device().is_cuda()) {
        img = img.to(torch::kCPU);
    }
    img = img.to(torch::kU8);

    // Make sure the tensor is contiguous and in the format [Height, Width, Channels]
    img = img.contiguous();

    cv::Mat mat(img.size(0), img.size(1), CV_8UC(img.size(2)), img.data_ptr<uchar>());

    // Display the image
    cv::imshow("Output Image", mat);
    cv::waitKey(1); // Wait for a key press (use 0 for infinite wait)


}

void CudaImplementation::updateCameraPose(glm::mat4 view, glm::mat4 proj, glm::vec3 pos) {
    // Inverting the first and third rows of the view matrix
    // Note: GLM is column-major, so we access columns via view[col][row]
    view[0][0] = -view[0][0];
    view[1][0] = -view[1][0];
    view[2][0] = -view[2][0];
    view[3][0] = -view[3][0];

    view[0][2] = -view[0][2];
    view[1][2] = -view[1][2];
    view[2][2] = -view[2][2];
    view[3][2] = -view[3][2];

    // Multiplying projection matrix with view matrix

    glm::mat4 projView = proj * view;
    torch::Device device(torch::kCUDA);
    viewmatrix = ConvertGlmMat4ToTensor(view).to(device);
    projmatrix = ConvertGlmMat4ToTensor(projView).to(device);
    campos     = ConvertGlmVec3ToTensor(pos).to(device);

}

void CudaImplementation::updateCameraIntrinsics(float hfox, float hfovy) {
    tan_fovx = hfox;
    tan_fovy = hfovy;
}


