//
// Created by magnus on 3/10/22.
//

#ifndef MULTISENSE_CRLCAMERAMODELS_H
#define MULTISENSE_CRLCAMERAMODELS_H


#include <MultiSense/external/glm/glm/glm.hpp>
#include <vulkan/vulkan_core.h>
#include <string>
#include <vector>
#include <utility>
#include  <memory>

#include <MultiSense/Src/Core/Texture.h>
#include <MultiSense/Src/Tools/Macros.h>
#include "include/MultiSense/MultiSenseTypes.hh"
#include <MultiSense/Src/Core/Definitions.h>

/***
 * @brief Bridges the gap between rendering vulkan images with data from MultiSense cameras. Tightly integrated with \refitem VkRender::TextureData
 */
class CRLCameraModels {

public:
    CRLCameraModels() = default;

    ~CRLCameraModels() {
        if (vulkanDevice != nullptr) {
            vkDestroyDescriptorSetLayout(vulkanDevice->logicalDevice, descriptorSetLayout, nullptr);
            vkDestroyDescriptorPool(vulkanDevice->logicalDevice, descriptorPool, nullptr);
            vkDestroyPipelineLayout(vulkanDevice->logicalDevice, pipelineLayout, nullptr);
            vkDestroyPipeline(vulkanDevice->logicalDevice, pipeline, nullptr);
            vkDestroyPipeline(vulkanDevice->logicalDevice, selectionPipeline, nullptr);
            vkDestroyPipelineLayout(vulkanDevice->logicalDevice, selectionPipelineLayout, nullptr);
        }
    }

    struct Model {
        explicit Model(const VkRender::RenderUtils *renderUtils);

        ~Model();

        /**@brief Property to flashing/disable drawing of this model. Set to false if you want to control when to draw the model. */
        bool draw = true;
        CRLCameraDataType modelType{};

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

        struct TextureIndices {
            uint32_t baseColor = -1;
            uint32_t normalMap = -1;
        };

        struct Dimensions {
            glm::vec3 min = glm::vec3(FLT_MAX);
            glm::vec3 max = glm::vec3(-FLT_MAX);
        } dimensions;

        VulkanDevice *vulkanDevice{};
        std::vector<std::string> extensions;

        std::unique_ptr<Texture2D> texture;
        std::unique_ptr<TextureVideo> textureVideo;
        std::unique_ptr<TextureVideo> textureColorMap;

        std::vector<Texture::TextureSampler> textureSamplers;
        TextureIndices textureIndices;

        void createMesh(VkRender::Vertex *_vertices, uint32_t vtxBufferSize);

        void setTexture(const std::basic_string<char, std::char_traits<char>, std::allocator<char>> &fileName);

        void
        createMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices,
                              const std::vector<uint32_t> &indices = std::vector<uint32_t>());

        void createEmtpyTexture(uint32_t width, uint32_t height, CRLCameraDataType texType);

        void updateTexture(CRLCameraDataType type);

        void setTexture(VkRender::TextureData *tex);

        void getTextureDataPointer(VkRender::TextureData *tex) const;

        void setZoom();
    };

    /**@brief Primitive for a surface */
    struct ImageData {
        struct {
            std::vector<VkRender::Vertex> vertices{};
            std::vector<uint32_t> indices;
            uint32_t vertexCount{};
            //uint32_t *indices{};
            uint32_t indexCount{};
        } quad{};

        /**@brief Generates a Quad with texture coordinates. Arguments are offset values */
        explicit ImageData(float x = 0.0f, float y = 0.0f) {
            int vertexCount = 4;
            int indexCount = 2 * 3;
            quad.vertexCount = vertexCount;
            quad.indexCount = indexCount;
            // Virtual class can generate some mesh data here
            quad.vertices.resize(vertexCount);
            quad.indices.resize(indexCount);

            auto *vP = quad.vertices.data();
            auto *iP = quad.indices.data();

            VkRender::Vertex vertex[4]{};
            vertex[0].pos = glm::vec3(-1.0f, -1.0f + y, 0.0f);
            vertex[1].pos = glm::vec3(1.0f, -1.0f + y, 0.0f);
            vertex[2].pos = glm::vec3(1.0f, 1.0f + y, 0.0f);
            vertex[3].pos = glm::vec3(-1.0f, 1.0f + y, 0.0f);

            vertex[0].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[1].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[2].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[3].normal = glm::vec3(0.0f, 1.0f, 0.0f);

            vertex[0].uv0 = glm::vec2(1.0f, 0.0f + y);
            vertex[1].uv0 = glm::vec2(0.0f, 0.0f + y);
            vertex[2].uv0 = glm::vec2(0.0f, 1.0f + y);
            vertex[3].uv0 = glm::vec2(1.0f, 1.0f + y);
            vP[0] = vertex[0];
            vP[1] = vertex[1];
            vP[2] = vertex[2];
            vP[3] = vertex[3];
            // indices
            iP[0] = 0;
            iP[1] = 1;
            iP[2] = 2;
            iP[3] = 2;
            iP[4] = 3;
            iP[5] = 0;
        }
    };

    std::vector<VkDescriptorSet> descriptors;
    VkDescriptorSetLayout descriptorSetLayout{};
    VkDescriptorPool descriptorPool{};
    VkPipeline pipeline{};
    VkPipeline selectionPipeline{}; // TODO destroy object
    bool initializedPipeline = false;

    VkPipelineLayout pipelineLayout{};
    VkPipelineLayout selectionPipelineLayout{};

    void destroy(VkDevice device);

    /**
     * Call to draw model
     * @param commandBuffer handle to commandbuffer to record drawing command
     * @param i index of swapchain image to render to
     * @param model modeol to draw
     * @param b if we want to render additional pipeline
     */
    void draw(VkCommandBuffer commandBuffer, uint32_t i, Model *model, bool b = true);

protected:

    const VulkanDevice *vulkanDevice = nullptr;

    /***
     * @brief Helper function
     * @param pModel
     */
    void createDescriptorSetLayout(Model *pModel);


    /**
     * @brief Helper function
     * Create the pipeline layout
     * @param pT pointer to store pipelinelayout object
     */
    void createPipelineLayout(VkPipelineLayout *pT);

    /**
     * @brief Bind a default descriptor layout to the pipeline for images
     * @param model Which model to configure
     * @param ubo reference to uniform buffers
     */
    void createImageDescriptors(Model *model, const std::vector<VkRender::UniformBufferSet> &ubo);

    /**
     * Bind a default descriptor layout fo the pipline for point clouds
     * @param model Which model to configure
     * @param ubo reference to uniform buffers
     */
    void createPointCloudDescriptors(Model *model, const std::vector<VkRender::UniformBufferSet> &ubo);

    /**
     * Create render pipeline
     * @param pT Handle to render pass
     * @param vector vector with shaders
     * @param type what kind of camera data we expect to handle
     * @param pPipelineT additional pipeline
     * @param pLayoutT additional pipeline layout
     */
    void createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> vector, CRLCameraDataType type,
                        VkPipeline *pPipelineT, VkPipelineLayout *pLayoutT);

    /**
     * Create descriptors for this model
     * @param count number of descriptorsets needed
     * @param ubo reference to uniform buffers
     * @param model which model to configure
     */
    void createDescriptors(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo, Model *model);

    /**
     * Creates the render pipeline using helper functions from this class
     * @param vector vector of shaders
     * @param model model to configure
     * @param renderUtils handle to render utilities from the scripts base \refitem Base class
     */
    void
    createRenderPipeline(const std::vector<VkPipelineShaderStageCreateInfo> &vector, Model *model,
                         const VkRender::RenderUtils *renderUtils);
};


#endif //MULTISENSE_CRLCAMERAMODELS_H
