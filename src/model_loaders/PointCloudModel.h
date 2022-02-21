//
// Created by magnus on 2/21/22.
//

#ifndef MULTISENSE_POINTCLOUDMODEL_H
#define MULTISENSE_POINTCLOUDMODEL_H

#include <utility>
#include <vulkan/vulkan_core.h>
#include <vector>
#include <string>
#include <MultiSense/src/core/Buffer.h>
#include <MultiSense/src/core/Texture.h>
#include <MultiSense/src/core/Base.h>
#include "glm/glm.hpp"
#include <MultiSense/src/tools/Macros.h>

class PointCloudModel {

public:
    PointCloudModel() = default;
    struct Model {
        struct Vertex {
            glm::vec3 pos;
            glm::vec3 normal;
            glm::vec2 uv0;
            glm::vec2 uv1;
            glm::vec4 joint0;
            glm::vec4 weight0;
        };

        struct UniformBufferSet {
            Buffer mvp;
            Buffer pointCloud;
        };


        struct Vertices {
            uint32_t count;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory;
        } vertices;
        struct Indices {
            uint32_t count;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory;
        } indices;

        std::vector<std::string> extensions;

        struct Dimensions {
            glm::vec3 min = glm::vec3(FLT_MAX);
            glm::vec3 max = glm::vec3(-FLT_MAX);
        } dimensions;

    } model;

    struct Dimensions {
        glm::vec3 min = glm::vec3(FLT_MAX);
        glm::vec3 max = glm::vec3(-FLT_MAX);
    } dimensions;


    std::vector<VkDescriptorSet> descriptors;
    VkDescriptorSetLayout descriptorSetLayout{};
    VkDescriptorPool descriptorPool{};
    VkPipeline pipeline{};
    VkPipelineLayout pipelineLayout{};

    glm::mat4 aabb{};
    std::vector<Texture> textures;
    std::vector<std::string> extensions;


    void destroy(VkDevice device);

    void loadFromFile(std::string filename, float scale = 1.0f);

    void createDescriptorSetLayout();

    void createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> vector);

    void createPipelineLayout();

    void draw(VkCommandBuffer commandBuffer, uint32_t i);


    void createDescriptors(uint32_t count, std::vector<Base::UniformBufferSet> ubo);

protected:

    VulkanDevice *vulkanDevice{};
    void transferDataStaging(Model::Vertex *_vertices, uint32_t vertexCount, unsigned int *_indices, uint32_t indexCount);

    void createRenderPipeline(const Base::RenderUtils &utils, std::vector<VkPipelineShaderStageCreateInfo> vector);

    void transferData(Model::Vertex *_vertices, uint32_t vertexCount, unsigned int *_indices, uint32_t indexCount);
};


#endif //MULTISENSE_POINTCLOUDMODEL_H
