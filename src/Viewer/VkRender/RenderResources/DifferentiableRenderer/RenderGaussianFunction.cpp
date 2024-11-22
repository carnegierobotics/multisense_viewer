//
// Created by magnus on 11/21/24.
//

#include "RenderGaussianFunction.h"


namespace VkRender::DR{
    torch::Tensor RenderGaussianFunction::forward(torch::autograd::AutogradContext *ctx,
                                                  torch::Tensor center,
                                                  torch::Tensor variance,
                                                  torch::Tensor xv,
                                                  torch::Tensor yv) {
        // Compute the difference between each pixel and the center
        torch::Tensor dx = xv - center[0];
        torch::Tensor dy = yv - center[1];

        // Adjust distance with per-axis variances
        torch::Tensor d_squared = (dx.pow(2) / variance[0]) + (dy.pow(2) / variance[1]);

        // Compute Gaussian function
        torch::Tensor img = torch::exp(-0.5 * d_squared);

        // Save tensors for backward pass
        ctx->save_for_backward({center, variance, dx, dy, d_squared, img});

        return img;
    }

    torch::autograd::tensor_list RenderGaussianFunction::backward(torch::autograd::AutogradContext *ctx,
                                                        torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor center = saved[0];
        torch::Tensor variance = saved[1];
        torch::Tensor dx = saved[2];
        torch::Tensor dy = saved[3];
        torch::Tensor d_squared = saved[4];
        torch::Tensor img = saved[5];


        // Return gradients with respect to inputs
        // The inputs were: center, variance, xv, yv
        // Since xv and yv do not require gradients, we return torch::Tensor()
        return {
                 torch::Tensor(),
                 torch::Tensor(),
                torch::Tensor(), // gradient w.r.t xv (None)
                torch::Tensor()  // gradient w.r.t yv (None)
        };
    }
}