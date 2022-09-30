//
// Created by magnus on 2/20/22.
//

#ifndef MULTISENSE_MESHMODEL_H
#define MULTISENSE_MESHMODEL_H


#include <MultiSense/external/glm/glm/glm.hpp>
#include <vulkan/vulkan_core.h>
#include <string>
#include <vector>
#include <MultiSense/src/Core/Texture.h>
#include <MultiSense/src/Tools/Macros.h>
#include <MultiSense/src/Scripts/Base.h>
/*
class MeshModel {

public:
    MeshModel() = default;

    struct Model {
        struct VideoTexture {
            std::vector<unsigned char *> pixels{};
            uint32_t imageSize = 0;
            uint32_t width = 0;
            uint32_t height = 0;
        }videos{};
        struct Vertex {
            glm::vec3 pos{};
            glm::vec3 normal{};
            glm::vec2 uv0{};
            glm::vec2 uv1{};
            glm::vec4 joint0{};
            glm::vec4 weight0{};
        };

        struct Primitive {
            uint32_t firstIndex{};
            uint32_t indexCount{};
            uint32_t vertexCount{};
            bool hasIndices{};
            //Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount);
            //void setBoundingBox(glm::vec3 min, glm::vec3 max);
        };

        struct Mesh {
            VulkanDevice *device;
            uint32_t firstIndex = 0;
            uint32_t indexCount = 0;
            uint32_t vertexCount = 0;
            bool hasIndices{};

            struct Vertices {
                VkBuffer buffer = VK_NULL_HANDLE;
                VkDeviceMemory memory{};
            } vertices{};
            struct Indices {
                int count;
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
        VulkanDevice *device{};
        std::vector<std::string> extensions;
        std::vector<Texture2D> textures;
        std::vector<TextureVideo> textureVideos;
        std::vector<Texture::TextureSampler> textureSamplers;
        TextureIndices textureIndices;

        void createMesh(Model::Vertex *_vertices, uint32_t vertexCount);

        void loadTextureSamplers();

        void setTexture(std::basic_string<char, std::char_traits<char>, std::allocator<char>> fileName);


        Model(uint32_t count, VulkanDevice *_vulkanDevice);

        void
        createMeshDeviceLocal(Vertex *_vertices, uint32_t vertexCount, unsigned int *_indices, uint32_t indexCount);

        void setVideoTexture(uint32_t frame);

        void prepareVideoTextures();
    };


    std::vector<VkDescriptorSet> descriptors;
    VkDescriptorSetLayout descriptorSetLayout{};
    VkDescriptorPool descriptorPool{};
    VkPipeline pipeline{};
    VkPipelineLayout pipelineLayout{};

    void destroy(VkDevice device);

    void loadFromFile(std::string filename, float scale = 1.0f);

    void createDescriptorSetLayout();

    void createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> vector, ScriptType type);

    void createPipelineLayout();

    void draw(VkCommandBuffer commandBuffer, uint32_t i, MeshModel::Model *model);

    void createDescriptors(uint32_t count, std::vector<Base::UniformBufferSet> ubo, MeshModel::Model *model);

protected:

    VulkanDevice *vulkanDevice{};

    void
    transferDataStaging(Model::Vertex *_vertices, uint32_t vertexCount, unsigned int *_indices, uint32_t indexCount);

    void createRenderPipeline(const Base::RenderUtils &utils, std::vector<VkPipelineShaderStageCreateInfo> vector,
                              Model *model,
                              ScriptType type);

};

*/
#endif //MULTISENSE_MESHMODEL_H
