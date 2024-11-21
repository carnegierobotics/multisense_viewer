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

        torch::Tensor grad_output = grad_outputs[0];

        // Gradients of the Gaussian function
        torch::Tensor d_img_dd_squared = -0.5 * img;  // ∂img/∂d_squared

        // Gradients of d_squared w.r.t dx, dy, variance
        torch::Tensor dd_squared_d_dx = 2 * dx / variance[0];  // ∂d_squared/∂dx
        torch::Tensor dd_squared_d_dy = 2 * dy / variance[1];  // ∂d_squared/∂dy
        torch::Tensor dd_squared_d_variance_x = -(dx.pow(2)) / (variance[0].pow(2));  // ∂d_squared/∂variance_x
        torch::Tensor dd_squared_d_variance_y = -(dy.pow(2)) / (variance[1].pow(2));  // ∂d_squared/∂variance_y

        // Gradients of the Gaussian function w.r.t dx, dy, and variances
        torch::Tensor grad_dx = grad_output * d_img_dd_squared * dd_squared_d_dx;
        torch::Tensor grad_dy = grad_output * d_img_dd_squared * dd_squared_d_dy;
        torch::Tensor grad_variance_x = grad_output * d_img_dd_squared * dd_squared_d_variance_x;
        torch::Tensor grad_variance_y = grad_output * d_img_dd_squared * dd_squared_d_variance_y;

        // Gradients w.r.t center and variance
        torch::Tensor grad_center = torch::stack({
                                                         -torch::sum(grad_dx),
                                                         -torch::sum(grad_dy)
                                                 });  // Shape: (2,)

        torch::Tensor grad_variance = torch::stack({
                                                           torch::sum(grad_variance_x),
                                                           torch::sum(grad_variance_y)
                                                   });  // Shape: (2,)

        // Return gradients with respect to inputs
        // The inputs were: center, variance, xv, yv
        // Since xv and yv do not require gradients, we return torch::Tensor()
        return {
                grad_center,
                grad_variance,
                torch::Tensor(), // gradient w.r.t xv (None)
                torch::Tensor()  // gradient w.r.t yv (None)
        };
    }
}