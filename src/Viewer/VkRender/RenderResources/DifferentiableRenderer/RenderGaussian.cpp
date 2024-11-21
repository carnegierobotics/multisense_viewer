//
// Created by magnus on 11/21/24.
//

#include "RenderGaussian.h"


namespace VkRender::DR {
    RenderGaussian::RenderGaussian(int64_t image_size_) : image_size(image_size_) {
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

        torch::Tensor xv_, yv_;


    }

    torch::Tensor RenderGaussian::forward(torch::Tensor center, torch::Tensor variance) {
        double min_variance = 1e-3;

        variance = torch::nn::functional::softplus(variance) + min_variance;

        last_rendered_image = RenderGaussianFunction::apply(
                center,
                variance,
                xv,
                yv
        );

        return last_rendered_image;
    }

    torch::Tensor RenderGaussian::get_rendered_image() {
        return last_rendered_image;
    }
}