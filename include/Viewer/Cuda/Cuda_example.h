//
// Created by mgjer on 15/01/2024.
//

#ifndef MULTISENSE_VIEWER_CUDA_EXAMPLE_H
#define MULTISENSE_VIEWER_CUDA_EXAMPLE_H


#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <memory>  // For std::unique_ptr

#include <torch/torch.h>

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
    explicit CudaImplementation(const RasterSettings *settings);

    void draw();

    void updateCameraIntrinsics(float hfox, float hfovy);

    void updateCameraPose(glm::mat4 view, glm::mat4 proj, glm::vec3 pos);

private:
    torch::Tensor means3D;
    torch::Tensor shs;
    torch::Tensor opacity;
    torch::Tensor scales;
    torch::Tensor rotations;
    torch::Tensor cov3D_precomp;
    torch::Tensor colors;
    torch::Tensor viewmatrix;
    torch::Tensor projmatrix;
    torch::Tensor campos;
    torch::Tensor bg;

    float scale_modifier;
    float tan_fovx;
    float tan_fovy;
    int image_height;
    int image_width;
    int degree;
    bool prefiltered;
    bool debug;
};

#endif //MULTISENSE_VIEWER_CUDA_EXAMPLE_H
