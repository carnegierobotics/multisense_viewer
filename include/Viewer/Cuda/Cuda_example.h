//
// Created by mgjer on 15/01/2024.
//

#ifndef MULTISENSE_VIEWER_CUDA_EXAMPLE_H
#define MULTISENSE_VIEWER_CUDA_EXAMPLE_H


#include <glm/mat4x4.hpp>

#include <torch/torch.h>
#include <driver_types.h>
#include "Viewer/Core/Texture.h"

class CudaImplementation {
public:
    struct RasterSettings {
        int imageWidth = 0, imageHeight = 0;
        float tanFovX = 0.0f;
        float tanFovY = 0.0f;
        int shDegree = 3;
        bool prefilter = false;
        bool debug = false;
        float scaleModifier = 1.0f;

        glm::mat4 viewMat;
        glm::mat4 projMat;
        glm::vec3 camPos;
    };
    CudaImplementation(VkInstance* instance, VkDevice device, const RasterSettings *settings, const std::filesystem::path& modelPath, uint32_t memSizeCuda,     std::vector<TextureCuda>* textures);
    void updateGaussianData();

    void draw(uint32_t i, void* streamToRun);

    void updateCameraIntrinsics(float hfox, float hfovy);

    void updateCameraPose(glm::mat4 view, glm::mat4 proj, glm::vec3 pos);
    void updateSettings(const RasterSettings& settings);

private:
    torch::Tensor means3D;
    torch::Tensor shs;
    torch::Tensor opacity;
    torch::Tensor scales;
    torch::Tensor rotations;
    torch::Tensor colors;
    torch::Tensor viewmatrix;
    torch::Tensor projmatrix;


    torch::Tensor campos;
    torch::Tensor bg;
    torch::Tensor cov3Dprecompute;

    float scale_modifier = 0;
    float tan_fovx = 0;
    float tan_fovy = 0;
    int image_height = 0;
    int image_width = 0;
    int degree = 0;
    bool prefiltered = 0;
    bool debug = 0;

    std::vector<void*> handles;
    std::vector<cudaExternalMemory_t> cudaExtMem;
    std::vector<void *> cudaMemPtr;
    std::vector<cudaMipmappedArray_t> cudaMipMappedArrays;
    std::vector<cudaArray_t> cudaFirstLevels;

};

#endif //MULTISENSE_VIEWER_CUDA_EXAMPLE_H
