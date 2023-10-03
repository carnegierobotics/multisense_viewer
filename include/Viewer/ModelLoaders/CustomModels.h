//
// Created by magnus on 10/3/23.
//

#ifndef MULTISENSE_VIEWER_CUSTOMMODELS_H
#define MULTISENSE_VIEWER_CUSTOMMODELS_H


#include "Viewer/Core/Definitions.h"
#include "Viewer/Scripts/Private/TextureDataDef.h"

class CustomModels {
private:
    struct Model {
        explicit Model(const VkRender::RenderUtils *renderUtils);

        ~Model();

        /**@brief Property to flashing/disable drawing of this m_Model. Set to false if you want to control when to draw the m_Model. */
        bool draw = true;
        CRLCameraDataType cameraDataType{};

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

            Buffer uniformBuffer{};

        } mesh{};

        struct Dimensions {
            glm::vec3 min = glm::vec3(FLT_MAX);
            glm::vec3 max = glm::vec3(-FLT_MAX);
        } dimensions;

        VulkanDevice *vulkanDevice{};
        std::vector<std::string> extensions;
        std::vector<Texture::TextureSampler> textureSamplers;

        void uploadMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices,
                                   const std::vector<uint32_t> &indices = std::vector<uint32_t>());

    };

    std::vector<VkDescriptorSet> descriptors;
    VkDescriptorSetLayout descriptorSetLayout{};
    VkDescriptorPool descriptorPool{};
    VkPipeline pipeline{};
    bool initializedPipeline = false;
    VkPipelineLayout pipelineLayout{};
    const VulkanDevice *vulkanDevice = nullptr;
    const VkRender::RenderUtils *renderer;
public:
    std::unique_ptr<Model> model;

    explicit CustomModels(const VkRender::RenderUtils *renderUtils) {
        renderer = renderUtils;
        vulkanDevice = renderUtils->device;
        model = std::make_unique<Model>(renderUtils);
    }

    ~CustomModels(){
        vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, descriptorPool, nullptr);
        vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, pipelineLayout, nullptr);
        vkDestroyPipeline(vulkanDevice->m_LogicalDevice, pipeline, nullptr);
    }

    void createDescriptorSetLayout();

    void createDescriptorPool();

    void createDescriptorSets();

    void createGraphicsPipeline(std::vector<VkPipelineShaderStageCreateInfo> vector);

    void draw(VkCommandBuffer commandBuffer, uint32_t cbIndex);
};


#endif //MULTISENSE_VIEWER_CUSTOMMODELS_H
