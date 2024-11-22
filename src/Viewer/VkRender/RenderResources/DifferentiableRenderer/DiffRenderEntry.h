//
// Created by magnus-desktop on 11/22/24.
//

#ifndef DIFFRENDERENTRY_H
#define DIFFRENDERENTRY_H

#include "Viewer/VkRender/RenderResources/DifferentiableRenderer/RenderGaussian.h"
#include "Viewer/VkRender/RenderResources/DifferentiableRenderer/Utils.h"

namespace VkRender::DR{
    class DiffRenderEntry {

      DiffRenderEntry();

      void setup();
      void update();

        RenderGaussian renderer = RenderGaussian(256);
        torch::Device device = torch::Device(torch::kCPU);

        bool iterateOnUpdate = true;

        struct OptimizerPackage {
            torch::Tensor initialCenter = torch::tensor({32.0, 32.0});
            torch::Tensor initialVariance = torch::tensor({2.0, 5.0});

            // Parameters with requires_grad = true
            torch::Tensor center = torch::tensor({32.0, 32.0}, torch::requires_grad());
            torch::Tensor variance = torch::tensor({2.0, 5.0}, torch::requires_grad());

            // Target position
            torch::Tensor targetCenter = torch::tensor({42.0, 48.0});
            torch::Tensor targetVariance = torch::tensor({25.0, 4.0});

            torch::Tensor targetImage;
            torch::Tensor initialImage;

            std::unique_ptr<torch::optim::Adam> optimizer;
            OptimizerPackage(RenderGaussian& renderer) {
                targetImage = renderer.forward(targetCenter, targetVariance);
                initialImage = renderer.forward(initialCenter, initialVariance);

                // Initialize optimizer with center and variance
                optimizer = std::make_unique<torch::optim::Adam>(
                    torch::optim::Adam(std::vector<torch::Tensor>{center, variance})
                );

                // Set learning rates for each parameter group
                {
                    // Get reference to the parameter groups
                    auto& param_groups = optimizer->param_groups();

                    // Set learning rate for 'center' parameter group (index 0)
                    auto& group0_options = static_cast<torch::optim::AdamOptions&>(param_groups[0].options());
                    group0_options.lr() = 0.25;

                    // Set learning rate for 'variance' parameter group (index 1)
                    auto& group1_options = static_cast<torch::optim::AdamOptions&>(param_groups[1].options());
                    group1_options.lr() = 0.1;
                }
            }

        }m_package;
    };

}



#endif //DIFFRENDERENTRY_H
