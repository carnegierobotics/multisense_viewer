//
// Created by magnus on 8/15/24.
//

#include "GaussianModelGraphicsPipeline.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"

namespace VkRender {


    GaussianModelGraphicsPipeline::GaussianModelGraphicsPipeline() {
        try {
            // Create a queue using the CPU device selector
            auto gpuSelector = [](const sycl::device &dev) {
                if (dev.is_gpu()) {
                    return 1; // Positive value to prefer GPU devices
                } else {
                    return -1; // Negative value to reject non-GPU devices
                }
            };

            auto cpuSelector = [](const sycl::device &dev) {
                if (dev.is_cpu()) {
                    return 1; // Positive value to prefer GPU devices
                } else {
                    return -1; // Negative value to reject non-GPU devices
                }
            };    // Define a callable device selector using a lambda

            queue = sycl::queue(useCPU ? cpuSelector : gpuSelector);
            // Use the queue for your computation
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->warning("GPU device not found");
            Log::Logger::getInstance()->info("Falling back to default device selector");
            // Fallback to default device selector
            queue = sycl::queue(sycl::property::queue::in_order());
        }


        Log::Logger::getInstance()->info("Selected Device {}",
                                         queue.get_device().get_info<sycl::info::device::name>().c_str());
    }

    template<>
    void
    GaussianModelGraphicsPipeline::bind<GaussianModelComponent>(
            GaussianModelComponent &modelComponent) {

    }


    void GaussianModelGraphicsPipeline::draw(CommandBuffer &cmdBuffers) {

    }
}