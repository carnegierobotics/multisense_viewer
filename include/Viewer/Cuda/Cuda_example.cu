//
// Created by mgjer on 15/01/2024.
//

#include <iostream>
#include <random>
#include <rasterize_points.h>

#include "Viewer/Cuda/Cuda_example.h"
#include "Viewer/Tools/Logger.h"
#include <Viewer/Core/Texture.h>

#include <torch/torch.h>
#include <Viewer/Tools/helper_cuda.h>


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

torch::Tensor ConvertGlmMat4ToTensor(const glm::mat4& mat) {
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

torch::Tensor ConvertGlmVec3ToTensor(const glm::vec3& vec) {
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

CudaImplementation::CudaImplementation(const RasterSettings* settings, std::vector<void*> handles) {
    this->handles = handles;

    cudaExtMem.resize(handles.size());
    cudaMemPtr.resize(handles.size());
    cudaMipMappedArrays.resize(handles.size());

    for (size_t i = 0; i < handles.size(); ++i) {
        cudaExternalMemoryHandleDesc cudaExtMemHandleDesc{};
        memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
        cudaExtMemHandleDesc.size = settings->imageHeight * settings->imageWidth * 4;
        cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        cudaExtMemHandleDesc.handle.win32.handle = handles[i];
        cudaExtMemHandleDesc.flags = 0;
        checkCudaErrors(cudaImportExternalMemory(&cudaExtMem[i], &cudaExtMemHandleDesc));

        // Step 3: CUDA memory copy
        /*
                cudaExternalMemoryBufferDesc bufferDesc{};
                memset(&bufferDesc, 0, sizeof(bufferDesc));
                bufferDesc.offset = 0;
                bufferDesc.size = cudaExtMemHandleDesc.size;
                checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&cudaMemPtr[i], cudaExtMem[i], &bufferDesc));
                */
        cudaExternalMemoryMipmappedArrayDesc desc = {};
        memset(&desc, 0, sizeof(desc));

        cudaChannelFormatDesc formatDesc;
        memset(&formatDesc, 0, sizeof(formatDesc));
        formatDesc.x = 8;
        formatDesc.y = 8;
        formatDesc.z = 8;
        formatDesc.w = 8;
        formatDesc.f = cudaChannelFormatKindUnsigned;

        cudaExtent extent = {0, 0, 0};
        extent.width = settings->imageWidth;
        extent.height = settings->imageHeight;
        extent.depth = 0;

        unsigned int flags = 0;
        flags |= cudaArrayLayered;
        flags |= cudaArrayColorAttachment;

        desc.offset = 0;
        desc.formatDesc = formatDesc;
        desc.extent = extent;
        desc.flags = 0;
        desc.numLevels = 1;

        checkCudaErrors(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipMappedArrays[i], cudaExtMem[i], &desc));
    }


    torch::Device device(torch::kCUDA);

    viewmatrix = ConvertGlmMat4ToTensor(settings->viewMat).to(device);
    projmatrix = ConvertGlmMat4ToTensor(settings->projMat).to(device);
    campos = ConvertGlmVec3ToTensor(settings->camPos).to(device);
    bg = torch::tensor({0.0, 0.0, 0.0}, torch::dtype(torch::kFloat32)).to(device);
    // Other parameters
    scale_modifier = settings->scaleModifier;
    tan_fovx = settings->tanFovX;
    tan_fovy = settings->tanFovY;
    image_height = settings->imageHeight;
    image_width = settings->imageWidth;
    degree = settings->shDegree;
    prefiltered = settings->prefilter;
    debug = settings->debug;

    auto gaussianData = naive_gaussian();
    // Example usage
    means3D = gaussianData.xyz.to(device);
    shs = gaussianData.sh.to(device);
    opacity = gaussianData.opacity.to(device);
    scales = gaussianData.scale.to(device);
    rotations = gaussianData.rot.to(device);
    cov3D_precomp = torch::tensor({}).to(device);
    colors = torch::tensor({}).to(device);

    degree = static_cast<int>(std::round(std::sqrt(gaussianData.sh_dim()))) - 1;
}

void CudaImplementation::updateGaussianData() {
    auto gaussianData = naive_gaussian();
    torch::Device device(torch::kCUDA);

    means3D = gaussianData.xyz.to(device);
    shs = gaussianData.sh.to(device);
    opacity = gaussianData.opacity.to(device);
    scales = gaussianData.scale.to(device);
    rotations = gaussianData.rot.to(device);
    cov3D_precomp = torch::tensor({}).to(device);
    colors = torch::tensor({}).to(device);

    degree = static_cast<int>(std::round(std::sqrt(gaussianData.sh_dim()))) - 1;
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

    proj[1][1] *= -1; // re-flip y-axis to match OpenGL and other impl.
    // Multiplying projection matrix with view matrix
    glm::mat4 projView = proj * view;
    torch::Device device(torch::kCUDA);
    viewmatrix = ConvertGlmMat4ToTensor(view).to(torch::kFloat).to(device);
    projmatrix = ConvertGlmMat4ToTensor(projView).to(torch::kFloat).to(device);
    campos = ConvertGlmVec3ToTensor(pos).to(torch::kFloat).to(device);
}

void CudaImplementation::updateSettings(const CudaImplementation::RasterSettings& settings) {
    scale_modifier = settings.scaleModifier;
}

void CudaImplementation::updateCameraIntrinsics(float hfox, float hfovy) {
    tan_fovx = hfox;
    tan_fovy = hfovy;
}

void CudaImplementation::printTensor(const torch::Tensor& tensor) {
    std::cout << "Shape/Size: " << tensor.sizes() << std::endl;
    std::cout << "Data Type: " << tensor.dtype() << std::endl;
    std::cout << "Device: " << tensor.device() << std::endl;
}

__global__ void convertFloatToUnorm(float* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 4; // 4 channels: RGBA
        output[idx] = static_cast<uint8_t>(input[idx] * 255);     // R
        output[idx + 1] = static_cast<uint8_t>(input[idx + 1] * 255); // G
        output[idx + 2] = static_cast<uint8_t>(input[idx + 2] * 255); // B
        output[idx + 3] = static_cast<uint8_t>(input[idx + 3] * 255); // A
    }
}

void CudaImplementation::draw(uint32_t i) {
    int rendered;
    torch::Tensor out_color, radii, geomBuffer, binningBuffer, imgBuffer;
    // Call the function
    std::tie(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer) = RasterizeGaussiansCUDA(
        bg, means3D, colors, opacity, scales, rotations,
        scale_modifier, cov3D_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy,
        image_height, image_width, shs, degree, campos, prefiltered, debug
    );
    auto img = out_color.permute({1, 2, 0}); // Change [Channels, Height, Width] to [Height, Width, Channels]
    img = img.contiguous();
    auto alpha_channel = torch::ones({img.size(0), img.size(1), 1}, img.options());
    auto img_with_alpha = torch::cat({img, alpha_channel}, 2);
    img_with_alpha = img_with_alpha.contiguous();
    size_t data_size = img_with_alpha.numel(); // Assuming the tensor is of type torch::kFloat
    //printTensor(img_with_alpha);

    cudaArray_t levelArray;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&levelArray, cudaMipMappedArrays[i], 0)); // 0 for the first level

    cudaMemcpy3DParms p = {0};
    p.srcPtr   = make_cudaPitchedPtr(img_with_alpha.data_ptr(), 1024 * sizeof(float), 1024, 1024);
    p.dstArray = levelArray;
    p.extent   = make_cudaExtent(1024, 1024, 1); // depth is 1 for 2D
    p.kind     = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&p));
    /*
    cudaMemcpy3DParms p = {0};
    p.srcPtr   = make_cudaPitchedPtr(img_with_alpha.data_ptr(), 1024 * sizeof(float), 1024, 1024);
    p.dstArray = levelArray;
    p.extent   = make_cudaExtent(width, height, depth); // depth is 1 for 2D
    p.kind     = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&p));
    */

    //checkCudaErrors(cudaMemcpy(cudaMipMappedArrays[i],img_with_alpha.data_ptr() , data_size, cudaMemcpyDeviceToDevice));
}
