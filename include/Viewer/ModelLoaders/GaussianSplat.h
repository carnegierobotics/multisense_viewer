//
// Created by mgjer on 10/01/2024.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANSPLAT_H
#define MULTISENSE_VIEWER_GAUSSIANSPLAT_H

#include <torch/torch.h>


namespace CUDARenderer {
    struct RasterSettings {

        RasterSettings()= default;
        uint32_t height = 0;
        uint32_t width = 0;
        float tanFovX = 0;
        float tanFovY = 0;
        torch::Tensor bg = torch::empty({0});
        float scaleModifier = 0;
        torch::Tensor viewMat = torch::empty({0});
        torch::Tensor projectionMat = torch::empty({0});
        float shDegree = 3;
        torch::Tensor cameraPosition = torch::empty({0});
        bool prefilter = false;
        bool debug = false;
    };

    struct GaussianData {
        torch::Tensor xyz;
        torch::Tensor rot;
        torch::Tensor scale;
        torch::Tensor opacity;
        torch::Tensor sh;

    };

};

class GaussianSplat {
    GaussianSplat() = default;
// Bind texture to be used by the rasterizer
    CUDARenderer::GaussianData gaussianData;
    CUDARenderer::RasterSettings rasterSettings;

    void cudaFromCpu(){
        torch::Tensor cuda_xyz = gaussianData.xyz.to(torch::kFloat).to(torch::kCUDA).set_requires_grad(false);
    }


};


#endif //MULTISENSE_VIEWER_GAUSSIANSPLAT_H
