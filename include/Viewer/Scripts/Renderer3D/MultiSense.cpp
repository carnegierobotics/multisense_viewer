//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/MultiSense.h"
#include "Viewer/ImGui/Widgets.h"

void MultiSense::setup() {
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {
        {
            loadShader("spv/object.vert",
                       VK_SHADER_STAGE_VERTEX_BIT)
        },
        {
            loadShader("spv/object.frag",
                       VK_SHADER_STAGE_FRAGMENT_BIT)
        }
    };

    KS21 = std::make_unique<GLTFModel::Model>(&renderUtils, renderUtils.device);
    KS21->loadFromFile(Utils::getAssetsPath().append("Models/s30_pbr.gltf").string(), renderUtils.device,
                       renderUtils.device->m_TransferQueue, 1.0f);
    KS21->createRenderPipeline(renderUtils, shaders);

    auto queue = sycl::queue(sycl::gpu_selector_v);

    auto device = queue.get_device();
    auto platform = device.get_platform();

    // Printing information about the platform
    std::cout << "Platform Info:" << std::endl;
    std::cout << "Platform Name: " << platform.get_info<sycl::info::platform::name>() << std::endl;
    std::cout << "Platform Vendor: " << platform.get_info<sycl::info::platform::vendor>() << std::endl;
    std::cout << "Platform Version: " << platform.get_info<sycl::info::platform::version>() << std::endl;

    // Printing information about the device
    std::cout << "\nDevice Info:" << std::endl;
    std::cout << "Device Name: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Device Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Device Version: " << device.get_info<sycl::info::device::version>() << std::endl;
    std::cout << "Driver Version: " << device.get_info<sycl::info::device::driver_version>() << std::endl;

    // Queue information is more about properties and capabilities rather than specific attributes
    // SYCL doesn't directly expose queue information like platform and device info
    std::cout << "\nQueue Info:" << std::endl;
    std::cout << "Queue Device Max Size: " << queue.get_device().get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "Queue is In Order: " << std::boolalpha << queue.is_in_order() << std::endl;


    startPlay = std::chrono::steady_clock::now();
}


void MultiSense::update() {
    std::chrono::duration<float> dt = std::chrono::steady_clock::now() - startPlay;

    auto& d = bufferOneData;
    d->model = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, 0.0f));
    d->model = glm::rotate(d->model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    // d->model = glm::scale(d->model, glm::vec3(0.1f, 0.1f, 0.1f));
    d->projection = renderData.camera->matrices.perspective;


    d->view = renderData.camera->matrices.view;
    d->camPos = glm::vec3(
        static_cast<double>(-renderData.camera->m_Position.z) * sin(
            static_cast<double>(glm::radians(renderData.camera->m_Rotation.y))) *
        cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.x))),
        static_cast<double>(-renderData.camera->m_Position.z) * sin(
            static_cast<double>(glm::radians(renderData.camera->m_Rotation.x))),
        static_cast<double>(renderData.camera->m_Position.z) *
        cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.y))) *
        cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.x)))
    );

    auto& d2 = bufferTwoData;
    d2->lightDir = glm::vec4(
        static_cast<double>(sinf(glm::radians(lightSource.rotation.x))) * cos(
            static_cast<double>(glm::radians(lightSource.rotation.y))),
        sin(static_cast<double>(glm::radians(lightSource.rotation.y))),
        cos(static_cast<double>(glm::radians(lightSource.rotation.x))) * cos(
            static_cast<double>(glm::radians(lightSource.rotation.y))),
        0.0f);


    auto* ptr = reinterpret_cast<VkRender::FragShaderParams*>(sharedData->data);
    d2->gamma = ptr->gamma;
    d2->exposure = ptr->exposure;
    d2->scaleIBLAmbient = ptr->scaleIBLAmbient;
    d2->debugViewInputs = ptr->debugViewInputs;
    d2->prefilteredCubeMipLevels = renderUtils.skybox.prefilteredCubeMipLevels;
}

void MultiSense::draw(CommandBuffer* commandBuffer, uint32_t i, bool b) {
    if (b) {
        KS21->draw(commandBuffer, i);
    }
}
