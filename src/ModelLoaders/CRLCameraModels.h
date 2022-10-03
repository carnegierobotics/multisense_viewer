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

#include <MultiSense/src/Core/Texture.h>
#include <MultiSense/src/Tools/Macros.h>
#include <MultiSense/src/Scripts/Base.h>
#include "include/MultiSense/MultiSenseTypes.hh"
#include <MultiSense/src/Core/Definitions.h>


class CRLCameraModels {

public:
    CRLCameraModels() = default;

    ~CRLCameraModels() {
        if (vulkanDevice) {
            vkDestroyDescriptorSetLayout(vulkanDevice->logicalDevice, descriptorSetLayout, nullptr);
            vkDestroyDescriptorPool(vulkanDevice->logicalDevice, descriptorPool, nullptr);
            vkDestroyPipelineLayout(vulkanDevice->logicalDevice, pipelineLayout, nullptr);
            vkDestroyPipeline(vulkanDevice->logicalDevice, pipeline, nullptr);
            vkDestroyPipeline(vulkanDevice->logicalDevice, selectionPipeline, nullptr);
            vkDestroyPipelineLayout(vulkanDevice->logicalDevice, selectionPipelineLayout, nullptr);
        }
    }

    struct Model {
        explicit Model(const Base::RenderUtils *renderUtils);
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
        std::unique_ptr<TextureVideo> textureVideoDepthMap;

        std::vector<Texture::TextureSampler> textureSamplers;
        TextureIndices textureIndices;

        void createMesh(VkRender::Vertex *_vertices, uint32_t vtxBufferSize);

        void setTexture(const std::basic_string<char, std::char_traits<char>, std::allocator<char>>& fileName);

        void
        createMeshDeviceLocal(VkRender::Vertex *_vertices, uint32_t vertexCount, unsigned int *_indices,
                              uint32_t indexCount);

        void createEmtpyTexture(uint32_t width, uint32_t height, CRLCameraDataType texType);

        void setTexture(VkRender::MP4Frame* frame);

        void setTexture(VkRender::YUVTexture *tex);

        void setTexture(VkRender::TextureData *tex);

        void setZoom();
    };

    /**@brief Primitive for a surface */
    struct ImageData {
        struct {
            void *vertices{};
            uint32_t vertexCount{};
            uint32_t *indices{};
            uint32_t indexCount{};
        } quad{};

        /**@brief Generates a Quad with texture coordinates. Arguments are offset values */
        ImageData(float x = 0.0f, float y = 0.0f) {
            int vertexCount = 4;
            int indexCount = 2 * 3;
            quad.vertexCount = vertexCount;
            quad.indexCount = indexCount;
            // Virtual class can generate some mesh data here
            quad.vertices = calloc(vertexCount, sizeof(VkRender::Vertex));
            quad.indices = static_cast<uint32_t *>(calloc(indexCount, sizeof(uint32_t)));

            auto *vP = (VkRender::Vertex *) quad.vertices;
            auto *iP = (uint32_t *) quad.indices;

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
    const Base::RenderUtils *utils = nullptr;
    bool initializedPipeline = false;

    VkPipelineLayout pipelineLayout{};
    VkPipelineLayout selectionPipelineLayout{};

    void destroy(VkDevice device);

    void loadFromFile(std::string filename, float scale = 1.0f);

    void createDescriptorSetLayout(Model *pModel);

    //void createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> vector, ScriptType type);

    void createPipelineLayout(VkPipelineLayout *pT);

    void draw(VkCommandBuffer commandBuffer, uint32_t i, Model *model, bool b = true);

    //void createDescriptors(uint32_t count, std::vector<Base::UniformBufferSet> ubo, CRLCameraModels::Model *model);

protected:

    VulkanDevice *vulkanDevice{};

    void createImageDescriptors(Model *model, const std::vector<Base::UniformBufferSet> &ubo);

    void createPointCloudDescriptors(Model *model, const std::vector<Base::UniformBufferSet> &ubo);

    void createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> vector, ScriptType type,
                        VkPipeline *pPipelineT, VkPipelineLayout *pLayoutT);

    void createDescriptors(uint32_t count, const std::vector<Base::UniformBufferSet> &ubo, Model *model);

    void
    createRenderPipeline(const std::vector<VkPipelineShaderStageCreateInfo>& vector, Model *model,
                         ScriptType type, const Base::RenderUtils *renderUtils);
};


#endif //MULTISENSE_CRLCAMERAMODELS_H
