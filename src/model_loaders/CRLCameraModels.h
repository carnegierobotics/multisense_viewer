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

#include <MultiSense/src/core/Texture.h>
#include <MultiSense/src/tools/Macros.h>
#include <MultiSense/src/core/Base.h>
#include "MultiSense/MultiSenseTypes.hh"
#include <MultiSense/src/crl_camera/CRLBaseCamera.h>

class CRLCameraModels {

public:
    CRLCameraModels() = default;

    struct Model {

        struct VideoTexture {
            std::vector<u_char *> pixels;
            VkDeviceSize imageSize;
            uint32_t width;
            uint32_t height;
        }videos;
        struct Vertex {
            glm::vec3 pos;
            glm::vec3 normal;
            glm::vec2 uv0;
            glm::vec2 uv1;
            glm::vec4 joint0;
            glm::vec4 weight0;
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
                VkDeviceMemory memory;
            } vertices;
            struct Indices {
                int count;
                VkBuffer buffer = VK_NULL_HANDLE;
                VkDeviceMemory memory;
            } indices;

            Buffer uniformBuffer;

        } mesh;

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

        void prepareTextureImage(uint32_t width, uint32_t height, VkDeviceSize size,
                                 CRLCameraDataType texType);

        void setVideoTexture(const crl::multisense::image::Header& streamOne, const crl::multisense::image::Header& streamTwo);
    };

    /**@brief Primitive for a surface */
    struct ImageData {
        struct {
            void *vertices{};
            uint32_t vertexCount{};
            uint32_t *indices{};
            uint32_t indexCount{};
        } quad;

        /**@brief Generates a Quad with texture coordinates */
        ImageData(float widthScale, float heightScale) {
            int vertexCount = 4;
            int indexCount = 2 * 3;
            quad.vertexCount = vertexCount;
            quad.indexCount = indexCount;
            // Virtual class can generate some mesh data here
            quad.vertices = calloc(vertexCount, sizeof(CRLCameraModels::Model::Vertex));
            quad.indices = static_cast<uint32_t *>(calloc(indexCount, sizeof(uint32_t)));

            auto *vP = (CRLCameraModels::Model::Vertex *) quad.vertices;
            auto *iP = (uint32_t *) quad.indices;

            CRLCameraModels::Model::Vertex vertex[4];
            vertex[0].pos = glm::vec3(0.0f, 0.0f, 0.0f);
            vertex[1].pos = glm::vec3(1.0f * widthScale, 0.0f, 0.0f);
            vertex[2].pos = glm::vec3(0.0f, 0.0f, 1.0f * heightScale);
            vertex[3].pos = glm::vec3(1.0f * widthScale, 0.0f, 1.0f * heightScale);

            vertex[0].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[1].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[2].normal = glm::vec3(0.0f, 1.0f, 0.0f);
            vertex[3].normal = glm::vec3(0.0f, 1.0f, 0.0f);

            vertex[0].uv0 = glm::vec2(0.0f, 0.0f);
            vertex[1].uv0 = glm::vec2(1.0f, 0.0f);
            vertex[2].uv0 = glm::vec2(0.0f, 1.0f);
            vertex[3].uv0 = glm::vec2(1.0f, 1.0f);
            vP[0] = vertex[0];
            vP[1] = vertex[1];
            vP[2] = vertex[2];
            vP[3] = vertex[3];
            // indices
            iP[0] = 0;
            iP[1] = 1;
            iP[2] = 2;
            iP[3] = 1;
            iP[4] = 2;
            iP[5] = 3;
        }
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

    void draw(VkCommandBuffer commandBuffer, uint32_t i, CRLCameraModels::Model *model);

    void createDescriptors(uint32_t count, std::vector<Base::UniformBufferSet> ubo, CRLCameraModels::Model *model);

protected:

    VulkanDevice *vulkanDevice{};

    void
    transferDataStaging(Model::Vertex *_vertices, uint32_t vertexCount, unsigned int *_indices, uint32_t indexCount);

    void createRenderPipeline(const Base::RenderUtils &utils, std::vector<VkPipelineShaderStageCreateInfo> vector,
                              Model *model,
                              ScriptType type);

};



#endif //MULTISENSE_CRLCAMERAMODELS_H
