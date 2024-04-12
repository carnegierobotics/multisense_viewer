//
// Created by magnus on 4/8/24.
//

#ifndef MULTISENSE_VIEWER_IMAGEVIEW_H
#define MULTISENSE_VIEWER_IMAGEVIEW_H

#include <stb_image.h>

#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/MultiSenseDeviceDefinitions.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Core/Texture.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/Scripts/Private/ScriptUtils.h"

class ImageView {

public:
    ImageView(const VkRender::RenderUtils *renderUtils, int width, int height, int channels,
              const std::vector<VkPipelineShaderStageCreateInfo> *shaders, bool useOffScreenImageRender) {

        VkRender::ScriptUtils::ImageData imgData(useOffScreenImageRender ? 0 : -2);
        m_shaders = shaders;
        m_renderUtils = renderUtils;
        m_model = std::make_unique<Model>(renderUtils->UBCount, renderUtils->device);

        assert(m_shaders != nullptr);
        m_vulkanDevice = renderUtils->device;
        m_model->createMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);
        // Load a default texture
        // Create texture m_Image if not created
        if (width == 0) {

            unsigned char *pixels = stbi_load((Utils::getTexturePath().append("moon.png")).string().c_str(),
                                              &width,
                                              &height,
                                              &channels,
                                              STBI_rgb_alpha);
            if (!pixels) {
                Log::Logger::getInstance()->error("Failed to load texture image {}",
                                                  (Utils::getTexturePath().append("no_source_selected.png")).string());
            }
            m_model->createEmptyTexture(width, height);

            for (uint32_t i = 0; i < renderUtils->UBCount; ++i) {
                auto *dataPtr = m_model->resources[0].texture[i]->m_DataPtr;
                std::memcpy(dataPtr, pixels, width * height * channels);
                m_model->resources[0].texture[i]->updateTextureFromBuffer();

                auto *dataPtr2 = m_model->resources[1].texture[i]->m_DataPtr;
                std::memcpy(dataPtr2, pixels, width * height * channels);
                m_model->resources[1].texture[i]->updateTextureFromBuffer();
            }
        } else
            m_model->createEmptyTexture(width, height);

        createDescriptors(useOffScreenImageRender);
        createGraphicsPipeline();
    };


    void draw(CommandBuffer *commandBuffer, uint32_t i);

    void updateTexture(uint32_t currentFrame, void *data, uint32_t size);

private:

    struct Model {
        explicit Model(uint32_t framesInFlight, VulkanDevice *vulkanDevice);


        VulkanDevice *m_vulkanDevice = nullptr;
        uint32_t m_framesInFlight = 1;

        struct Resource {
            VkDescriptorPool descriptorPool{};
            std::vector<VkDescriptorSet> descriptors;
            std::vector<VkDescriptorSetLayout> descriptorSetLayout;
            std::vector<VkPipeline> pipeline;
            std::vector<VkPipelineLayout> pipelineLayout;
            std::vector<std::unique_ptr<TextureVideo>> texture;
        };

        std::vector<Resource> resources;


        struct Mesh {
            VulkanDevice *device = nullptr;
            uint32_t firstIndex = 0;
            uint32_t indexCount = 0;
            uint32_t vertexCount = 0;
            struct Vertices {
                VkBuffer buffer = VK_NULL_HANDLE;
                VkDeviceMemory memory{};
            };
            struct Indices {
                VkBuffer buffer = VK_NULL_HANDLE;
                VkDeviceMemory memory{};
            };
            std::vector<Vertices> vertices;
            std::vector<Indices> indices;
            Buffer uniformBuffer{};

        } m_mesh{};

        void createMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices,
                                   const std::vector<uint32_t> &indices = std::vector<uint32_t>());

        void createEmptyTexture(uint32_t width, uint32_t height);

        ~Model();
    };


    void createDescriptors(bool useOffScreenImageRender);

    void createGraphicsPipeline();

    const VkRender::RenderUtils *m_renderUtils;
    VulkanDevice *m_vulkanDevice = nullptr;
    std::unique_ptr<Model> m_model;
    const std::vector<VkPipelineShaderStageCreateInfo> *m_shaders;

};


#endif //MULTISENSE_VIEWER_IMAGEVIEW_H
