#ifndef DIFFRENDERENTRY_H
#define DIFFRENDERENTRY_H

#include <torch/torch.h>
#include "Viewer/VkRender/RenderResources/DifferentiableRenderer/RenderGaussian.h"
#include "Viewer/VkRender/RenderResources/DifferentiableRenderer/Utils.h"

namespace VkRender::DR {

    class DiffRenderEntry {
    public:
        DiffRenderEntry();

        // Setup method to initialize the renderer and optimizer
        void setup();

        // Update method to perform updates per frame or as needed
        void update();

        // Iterate method to perform an optimization step
        void iterate();

        // Method to access the gradient of the parameters
        torch::Tensor getGradient() const;

        // Set new parameters for optimization
        void setParameters(const torch::Tensor &center, const torch::Tensor &variance);

        // Get current parameters
        std::pair<torch::Tensor, torch::Tensor> getParameters() const;

        // Enable or disable iteration on update
        void setIterateOnUpdate(bool iterate);

        torch::Tensor getImage() {
            if (renderer.getLastRenderedImage().defined())
                return renderer.getLastRenderedImage().clone().detach().contiguous().cpu();
            return torch::zeros(renderer.imageSize());
        }

        uint32_t getImageSize(){
            return renderer.imageSize();
        }

    private:
        RenderGaussian renderer;
        torch::Device device;

        bool iterateOnUpdate = true;
        uint32_t m_iteration = 0;

        struct OptimizerPackage {
            // Initial parameters
            torch::Tensor initialCenter;
            torch::Tensor initialVariance;

            // Parameters with requires_grad = true
            torch::Tensor center;
            torch::Tensor variance;

            // Target parameters
            torch::Tensor targetCenter;
            torch::Tensor targetVariance;

            // Rendered images
            torch::Tensor targetImage;
            torch::Tensor initialImage;

            // Optimizer
            std::unique_ptr<torch::optim::Adam> optimizer;

            OptimizerPackage(RenderGaussian &renderer, torch::Device device = torch::kCPU);

            // Move tensors to the specified device
            void to(torch::Device device = torch::kCPU);
        };

        std::unique_ptr<OptimizerPackage> m_package;
    };

} // namespace VkRender::DR

#endif // DIFFRENDERENTRY_H
