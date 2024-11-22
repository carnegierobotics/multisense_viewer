//
// Created by magnus-desktop on 11/22/24.
//

#include "Viewer/VkRender/RenderResources/DifferentiableRenderer/DiffRenderEntry.h"

#include <cstdint>
#include <torch/torch.h>


namespace VkRender::DR {
    DiffRenderEntry::DiffRenderEntry(): m_package(renderer) {
        // Device configuration
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
        }
    }

    void DiffRenderEntry::setup() {

        // Enable anomaly detection
        torch::autograd::AnomalyMode::set_enabled(true);
        // Optimizer with different learning rates for center and variance

    }

    void DiffRenderEntry::update() {

        if (iterateOnUpdate) {
            m_package.optimizer->zero_grad();

            // Render the image with the current circle position
            torch::Tensor rendered_image = renderer.forward(m_package.center, m_package.variance);

            // Compute loss
            double weight = 0.2;
            torch::Tensor loss = Utils::L1Loss(rendered_image, m_package.targetImage);
            //torch::Tensor ssim_val = loss_utils::ssim(rendered_image, target_image);
            //torch::Tensor loss = (1.0 - weight) * Ll1 + weight * (1.0 - ssim_val);

            // Backpropagation
            loss.backward();

            // Update parameters
            m_package.optimizer->step();

            // Clamp variance
            {
                //torch::NoGradGuard no_grad;
                //m_package.variance.clamp_(1e-3);
            }
        }
    }
}
