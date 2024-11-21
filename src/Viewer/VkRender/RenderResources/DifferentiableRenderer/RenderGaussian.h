//
// Created by magnus on 11/21/24.
//

#ifndef MULTISENSE_VIEWER_RENDERGAUSSIAN_H
#define MULTISENSE_VIEWER_RENDERGAUSSIAN_H

#include <torch/torch.h>
#include "Viewer/VkRender/RenderResources/DifferentiableRenderer/RenderGaussianFunction.h"


namespace VkRender::DR{
    class RenderGaussian : public torch::nn::Module {
    public:
        RenderGaussian(int64_t image_size);

        torch::Tensor forward(torch::Tensor center, torch::Tensor variance);

        torch::Tensor get_rendered_image();

    private:
        int64_t image_size;
        torch::Tensor xv;
        torch::Tensor yv;
        torch::Tensor last_rendered_image;  // Store the last rendered image
    };

}


#endif //MULTISENSE_VIEWER_RENDERGAUSSIAN_H
