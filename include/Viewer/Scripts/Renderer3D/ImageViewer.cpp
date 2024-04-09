//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/ImageViewer.h"
#include "Viewer/ImGui/Widgets.h"

//#include <sycl/sycl.hpp>

void ImageViewer::setup() {

    //auto queue = sycl::queue(sycl::gpu_selector_v);
    //auto device = queue.get_device();
    //auto platform = device.get_platform();
    //// Printing information about the platform
    //std::cout << "Platform Info:" << std::endl;
    //std::cout << "Platform Name: " << platform.get_info<sycl::info::platform::name>() << std::endl;
    //std::cout << "Platform Vendor: " << platform.get_info<sycl::info::platform::vendor>() << std::endl;
    //std::cout << "Platform Version: " << platform.get_info<sycl::info::platform::version>() << std::endl;
    //// Printing information about the device
    //std::cout << "\nDevice Info:" << std::endl;
    //std::cout << "Device Name: " << device.get_info<sycl::info::device::name>() << std::endl;
    //std::cout << "Device Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
    //std::cout << "Device Version: " << device.get_info<sycl::info::device::version>() << std::endl;
    //std::cout << "Driver Version: " << device.get_info<sycl::info::device::driver_version>() << std::endl;
    //// Queue information is more about properties and capabilities rather than specific attributes
    //// SYCL doesn't directly expose queue information like platform and device info
    //std::cout << "\nQueue Info:" << std::endl;
    //std::cout << "Queue Device Max Size: " << queue.get_device().get_info<sycl::info::device::max_work_group_size>() << std::endl;
    //std::cout << "Queue is In Order: " << std::boolalpha << queue.is_in_order() << std::endl;


    // Create a Quad Texture
    // - Will change very often
    // - Determine where to display on the screen
    // - Very similar to CRLCameraModels but should be minimal
    uint32_t width = 300, height = 300, channels = 4;

    std::string vertexShaderFileName = "spv/default.vert";
    std::string fragmentShaderFileName = "spv/default.frag";
    VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};



    imageView = std::make_unique<ImageView>(&renderUtils, width, height, channels, &shaders, true);


}


void ImageViewer::update() {

    auto& d = bufferOneData;
    d->model = glm::mat4(1.0f);

    float xOffsetPx = (renderData.width - 150.0) / renderData.width;

    float translationX = xOffsetPx * 2 - 1;
    float translationY = xOffsetPx * 2 - 1;

    // Apply translation after scaling
    d->model = glm::translate(d->model, glm::vec3(translationX, translationY, 0.0f));
    // Convert 300 pixels from the right edge into NDC
    float scaleX = 300.0f / renderData.width;
    float scaleY = 0;

    d->model = glm::scale(d->model, glm::vec3(scaleX, scaleX, 1.0f)); // Uniform scaling in x and y

    //auto* pixels = (uint8_t*) malloc(1920 * 1080 * 4);
    //for(int i = 0; i < 1920 * 1080 * 4; ++i){
    //    float noise = static_cast<float>(random()) / static_cast<float>(RAND_MAX); // Generate a value between 0.0 and 1.0
    //    pixels[i] = static_cast<uint8_t>(noise * 255); // Scale to 0-255 and convert to uint8_t
    //}


    //imageView->updateTexture(renderData.index, nullptr, 1920 * 1080 * 4);

    //free(pixels);
}

void ImageViewer::draw(CommandBuffer* commandBuffer, uint32_t i, bool b) {
    if (commandBuffer->boundRenderPass == "main"){
        if (b){
            imageView->draw(commandBuffer, i);
        }
    }
}
