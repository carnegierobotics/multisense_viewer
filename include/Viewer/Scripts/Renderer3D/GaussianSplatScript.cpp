//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/GaussianSplatScript.h"


#include <Viewer/ImGui/Widgets.h>

#ifdef WIN32
#define FD_HANDLE HANDLE
#include <vulkan/vulkan_win32.h>
#include <AclAPI.h>

#else
#define FD_HANDLE int
#endif

void GaussianSplatScript::setup() {
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {
        {
            loadShader("spv/default.vert",
                       VK_SHADER_STAGE_VERTEX_BIT)
        },
        {
            loadShader("spv/default.frag",
                       VK_SHADER_STAGE_FRAGMENT_BIT)
        }
    };
    uniformBuffers.resize(renderUtils.UBCount);
    textures.resize(renderUtils.UBCount);
    // Create texture m_Image if not created
    int texWidth = 1280, texHeight = 720, texChannels = 0;
    auto* pixels = malloc(texWidth * texHeight * 16);
    PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetInstanceProcAddr(*renderUtils.instance, "vkGetMemoryWin32HandleKHR"));
    if (fpGetMemoryWin32HandleKHR == nullptr) {
        Log::Logger::getInstance()->error("Function not available");
    }
    handles.resize(textures.size());

    for (size_t i = 0; i < renderUtils.UBCount; ++i) {
        renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         &uniformBuffers[i], sizeof(VkRender::UBOMatrix));
        uniformBuffers[i].map();

        textures[i].fromBuffer(pixels, texWidth * texHeight * 16, VK_FORMAT_R32G32B32A32_SFLOAT, texWidth, texHeight,
                               renderUtils.device,
                               renderUtils.device->m_TransferQueue);

        VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
        vkMemoryGetWin32HandleInfoKHR.sType =
            VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
        vkMemoryGetWin32HandleInfoKHR.memory = textures[i].m_DeviceMemory;
        vkMemoryGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        if (fpGetMemoryWin32HandleKHR(renderUtils.device->m_LogicalDevice, &vkMemoryGetWin32HandleInfoKHR,
                                      &handles[i]) !=
            VK_SUCCESS) {
            Log::Logger::getInstance()->error("vkGetMemoryWin32HandleKHR not available");
        }
    }
    stbi_image_free(pixels);

    // setup descriptors and such
    RenderResource::MeshConfig meshConf;
    meshConf.device = renderUtils.device;
    mesh = std::make_unique<RenderResource::Mesh>(meshConf);

    RenderResource::PipelineConfig pConf;
    pConf.device = renderUtils.device;
    pConf.shaders = &shaders;
    pConf.textures = &textures;
    pConf.UboCount = renderUtils.UBCount;
    pConf.msaaSamples = renderUtils.msaaSamples;
    pConf.renderPass = renderUtils.renderPass;
    pConf.ubo = uniformBuffers.data();
    pipeline = std::make_unique<RenderResource::Pipeline>(pConf);


    /*
    splat = std::make_unique<GaussianSplat>(renderUtils.device);
    int device = splat->setCudaVkDevice(renderUtils.vkDeviceUUID);
    cudaStream_t streamToRun;
    checkCudaErrors(cudaStreamCreate(&streamToRun));
    */
    auto camParams = renderData.camera->getFocalParams(texWidth, texHeight);

    settings.camPos = renderData.camera->m_Position;
    settings.viewMat = renderData.camera->matrices.view;
    settings.projMat = renderData.camera->matrices.perspective;
    settings.imageWidth = texWidth;
    settings.imageHeight = texHeight;
    settings.shDegree = 3;
    settings.tanFovY = camParams.htany;
    settings.tanFovX = camParams.htanx;



    Widgets::make()->text(WIDGET_PLACEMENT_RENDERER3D, "Set scale modifier");
    Widgets::make()->slider(WIDGET_PLACEMENT_RENDERER3D, "##scale modifier", &scaleModifier, 0.1f, 5.0f);
    /*
    Widgets::make()->text(WIDGET_PLACEMENT_RENDERER3D, "Set camera pos");
    Widgets::make()->vec3(WIDGET_PLACEMENT_RENDERER3D, "##camera pos", &cameraPos);

    Widgets::make()->text(WIDGET_PLACEMENT_RENDERER3D, "Set camera target");
    Widgets::make()->vec3(WIDGET_PLACEMENT_RENDERER3D, "##camera target", &target);

    Widgets::make()->text(WIDGET_PLACEMENT_RENDERER3D, "Set camera up");
    Widgets::make()->vec3(WIDGET_PLACEMENT_RENDERER3D, "##camera up", &up);
    */

    Widgets::make()->fileDialog(WIDGET_PLACEMENT_RENDERER3D, "load model", &plyFileFolder);

    cudaImplementation = std::make_unique<CudaImplementation>(&settings, handles, filePathDialog);

}


void GaussianSplatScript::update() {
    if (plyFileFolder.valid()) {
        if (plyFileFolder.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            std::string selectedFolder = plyFileFolder.get(); // This will also make the future invalid
            if (!selectedFolder.empty()) {
                // Do something with the selected folder
                cudaImplementation = std::make_unique<CudaImplementation>(&settings, handles, selectedFolder);

            }
        }
    }

    mvpMat.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    mvpMat.projection = renderData.camera->matrices.perspective;
    mvpMat.view = renderData.camera->matrices.view;
    memcpy(uniformBuffers[renderData.index].mapped, &mvpMat, sizeof(VkRender::UBOMatrix));

    settings.scaleModifier = scaleModifier;
    cudaImplementation->updateCameraPose(renderData.camera->matrices.view, renderData.camera->matrices.perspective,
                                         renderData.camera->m_Target);
    cudaImplementation->updateSettings(settings);
}

void GaussianSplatScript::draw(CommandBuffer* commandBuffer, uint32_t i, bool b) {
    if (b) {
        cudaImplementation->draw(i);

        vkCmdBindDescriptorSets(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline->data.pipelineLayout, 0,
                                1,
                                &pipeline->data.descriptors[i], 0, nullptr);
        vkCmdBindPipeline(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->data.pipeline);
        const VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(commandBuffer->buffers[i], 0, 1, &mesh->model.vertices.buffer, offsets);
        //vkCmdDraw(commandBuffer->buffers[i], 6, 1, 0, 0);
        if (mesh->model.indexCount) {
            vkCmdBindIndexBuffer(commandBuffer->buffers[i], mesh->model.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer->buffers[i], mesh->model.indexCount, 1, mesh->model.firstIndex, 0, 0);
        }
        else {
        }
    }
}

void GaussianSplatScript::onDestroy() {
    Base::onDestroy();
}
