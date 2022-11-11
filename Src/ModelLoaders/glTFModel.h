//
// Created by magnus on 12/10/21.
//

#ifndef MULTISENSE_GLTFMODEL_H
#define MULTISENSE_GLTFMODEL_H


#include <vulkan/vulkan_core.h>
#include <MultiSense/external/glm/glm/detail/type_quat.hpp>
#include "MultiSense/Src/Core/VulkanDevice.h"
#include "glm/glm.hpp"
#include "MultiSense/Src/Core/Texture.h"

#include <MultiSense/external/glm/glm/gtc/type_ptr.hpp>
#include <MultiSense/Src/Tools/Macros.h>

// Changing this value here also requires changing it in the vertex shader
#define MAX_NUM_JOINTS 128u

class glTFModel {
public:

    glTFModel();
    ~glTFModel() = default;
    VulkanDevice *vulkanDevice = nullptr;

    struct Primitive {
        uint32_t m_FirstIndex{};
        uint32_t m_IndexCount{};
        Primitive(uint32_t _firstIndex, uint32_t indexCount);
    };

    struct Mesh {
        VulkanDevice *device;
        std::vector<Primitive*> primitives;
        struct UniformBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
            VkDescriptorBufferInfo descriptor;
            VkDescriptorSet descriptorSet;
            void *mapped;
        } uniformBuffer{};

    } m_Mesh;

    struct Node {
        Node *parent;
        uint32_t index;
        std::vector<Node*> children;
        glm::mat4 matrix;
        std::string name;
        Mesh *mesh;
        int32_t skinIndex = -1;
        glm::vec3 translation{ 0.0f};
        glm::vec3 scale{ 1.0f };
        glm::quat rotation{};
        glm::mat4 localMatrix();
        glm::mat4 getMatrix();
        void update();
        ~Node() = default;
    };

    struct Skin {
        std::string name;
        Node *skeletonRoot = nullptr;
        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<Node*> joints;
    };

    struct Material {
        enum AlphaMode{ ALPHAMODE_OPAQUE, ALPHAMODE_MASK, ALPHAMODE_BLEND };
        AlphaMode alphaMode = ALPHAMODE_OPAQUE;
        float alphaCutoff = 1.0f;
        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        glm::vec4 emissiveFactor = glm::vec4(1.0f);
        Texture2D *baseColorTexture;
        Texture2D *normalTexture;
        struct TexCoordSets {
            uint8_t baseColor = 0;
            uint8_t metallicRoughness = 0;
            uint8_t specularGlossiness = 0;
            uint8_t normal = 0;
            uint8_t occlusion = 0;
            uint8_t emissive = 0;
        } texCoordSets;
        struct PbrWorkflows {
            bool metallicRoughness = true;
            bool specularGlossiness = false;
        } pbrWorkflows;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    };

    struct TextureIndices{
        int baseColor = -1;
        int normalMap = -1;
    } ;

    struct Model {
        VulkanDevice *vulkanDevice = nullptr;
        std::string m_FileName;
        explicit Model(VulkanDevice* dev){
            this->vulkanDevice = dev;
        }
        ~Model();

        VulkanDevice *m_Device;
        std::vector<Skin*> skins;
        std::vector<std::string> extensions;
        std::vector<Primitive> primitives;
        std::vector<Texture2D> textures;
        std::vector<Material> materials;
        std::vector<Texture::TextureSampler> textureSamplers;
        TextureIndices textureIndices;
        bool useCustomTranslation = false;
        glm::vec3 nodeTranslation{};
        glm::vec3 nodeScale = glm::vec3(1.0f, 1.0f, 1.0f);

        struct Vertex {
            glm::vec3 pos;
            glm::vec3 normal;
            glm::vec2 uv0;
            glm::vec2 uv1;
            glm::vec4 joint0;
            glm::vec4 weight0;
        };

        struct Vertices {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory;
        } vertices{};
        struct Indices {
            int count = 0;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory;
        } indices{};


        std::vector<Node*> nodes{};
        std::vector<Node*> linearNodes{};

        struct Dimensions {
            glm::vec3 min = glm::vec3(FLT_MAX);
            glm::vec3 max = glm::vec3(-FLT_MAX);
        } dimensions{};

        void destroy(VkDevice device);
        void loadNode(glTFModel::Node* parent, const tinygltf::Node& node, uint32_t nodeIndex, const tinygltf::Model& _model, std::vector<uint32_t>& indexBuffer, std::vector<Vertex>& vertexBuffer, float globalscale);
        void loadSkins(tinygltf::Model& gltfModel);
        void loadTextures(tinygltf::Model& gltfModel, VulkanDevice* device, VkQueue transferQueue);
        VkSamplerAddressMode getVkWrapMode(int32_t wrapMode);
        VkFilter getVkFilterMode(int32_t filterMode);
        void loadTextureSamplers(tinygltf::Model& gltfModel);
        void loadMaterials(tinygltf::Model& gltfModel);
        void loadFromFile(std::string fileName, VulkanDevice *device, VkQueue transferQueue, float scale);
        Node* findNode(Node* parent, uint32_t index);
        Node* nodeFromIndex(uint32_t index);

        void setTexture(std::basic_string<char, std::char_traits<char>, std::allocator<char>> basicString);
        void setNormalMap(std::basic_string<char, std::char_traits<char>, std::allocator<char>> basicString);
        std::vector<VkDescriptorSet> descriptors;
        VkDescriptorSetLayout descriptorSetLayout{};
        VkDescriptorSetLayout descriptorSetLayoutNode{};

        VkDescriptorPool descriptorPool{};
        VkPipeline pipeline{};
        VkPipelineLayout pipelineLayout{};


        void createDescriptorSetLayout();

        void createDescriptors(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo);

        void setupNodeDescriptorSet(Node *node);

        void createPipeline(VkRenderPass renderPass, std::vector<VkPipelineShaderStageCreateInfo> shaderStages);


        void draw(VkCommandBuffer commandBuffer, uint32_t i);
        void drawNode(Node *node, VkCommandBuffer commandBuffer);

        void createRenderPipeline(const VkRender::RenderUtils& utils, const std::vector<VkPipelineShaderStageCreateInfo>& shaders);


        void translate(const glm::vec3 &translation);
        void scale(const glm::vec3 &scale);

        void createDescriptorsAdditionalBuffers(const std::vector<VkRender::RenderDescriptorBuffersData> &ubo);

        void
        createRenderPipeline(const VkRender::RenderUtils &utils,
                             const std::vector<VkPipelineShaderStageCreateInfo> &shaders,
                             const std::vector<VkRender::RenderDescriptorBuffersData> &buffers, ScriptType flags);

        void createDescriptorSetLayoutAdditionalBuffers();
    };



};


#endif //MULTISENSE_GLTFMODEL_H