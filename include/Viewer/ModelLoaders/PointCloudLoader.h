//
// Created by magnus on 10/9/23.
//

#ifndef MULTISENSE_VIEWER_POINTCLOUDLOADER_H
#define MULTISENSE_VIEWER_POINTCLOUDLOADER_H

#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/Scripts/Private/TextureDataDef.h"

class PointCloudLoader {
public:
    struct Model {
        explicit Model(const VkRender::RenderUtils *renderUtils);

        ~Model();

        /**@brief Property to flashing/disable drawing of this m_Model. Set to false if you want to control when to draw the m_Model. */
        bool draw = true;

        struct Mesh {
            VulkanDevice *device = nullptr;
            uint32_t firstIndex = 0;
            uint32_t indexCount = 0;
            uint32_t vertexCount = 0;

            struct Vertices {
                VkBuffer buffer = VK_NULL_HANDLE;
                VkDeviceMemory memory{};
            } vertices{};
            struct Indices {
                VkBuffer buffer = VK_NULL_HANDLE;
                VkDeviceMemory memory{};
            } indices{};

        } mesh{};

        VulkanDevice *vulkanDevice{};
        Texture2D disparityTexture;
        Texture2D colorTexture;

        void
        createMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices);
        void updateTexture();

        void createTexture(uint32_t width, uint32_t height);
    };

    std::vector<VkDescriptorSet> descriptors;
    VkDescriptorSetLayout descriptorSetLayout{};
    VkDescriptorPool descriptorPool{};
    VkPipeline pipeline{};
    VkPipelineLayout pipelineLayout{};
    const VulkanDevice *vulkanDevice = nullptr;
    const VkRender::RenderUtils *renderer;
public:
    std::unique_ptr<Model> model;
    std::vector<VkBuffer>* buffers = nullptr;

    explicit PointCloudLoader(const VkRender::RenderUtils *renderUtils) {
        renderer = renderUtils;
        vulkanDevice = renderUtils->device;
        model = std::make_unique<Model>(renderUtils);
    }

    ~PointCloudLoader(){
        vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, descriptorPool, nullptr);
        vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, pipelineLayout, nullptr);
        vkDestroyPipeline(vulkanDevice->m_LogicalDevice, pipeline, nullptr);
    }

    void createDescriptorSetLayout();

    void createDescriptorPool();

    void createDescriptorSets();

    void createGraphicsPipeline(std::vector<VkPipelineShaderStageCreateInfo> vector);

    void draw(CommandBuffer * commandBuffer, uint32_t cbIndex);

};


#endif //MULTISENSE_VIEWER_POINTCLOUDLOADER_H
