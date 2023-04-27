/**
 * @file: MultiSense-Viewer/include/Viewer/ModelLoaders/GLTFModel.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2021-12-10, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_GLTFMODEL_H
#define MULTISENSE_GLTFMODEL_H


#include <vulkan/vulkan_core.h>
#include <glm/glm/detail/type_quat.hpp>
#include <glm/glm.hpp>

#include "Viewer/Tools/Macros.h"
#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Texture.h"


class GLTFModel {
public:

    GLTFModel();
    ~GLTFModel() = default;
    VulkanDevice *vulkanDevice = nullptr;

    enum PBRWorkflows{ PBR_WORKFLOW_METALLIC_ROUGHNESS = 0, PBR_WORKFLOW_SPECULAR_GLOSINESS = 1 };


    struct PushConstBlockMaterial {
        glm::vec4 baseColorFactor;
        glm::vec4 emissiveFactor;
        glm::vec4 diffuseFactor;
        glm::vec4 specularFactor;
        float workflow;
        int colorTextureSet;
        int PhysicalDescriptorTextureSet;
        int normalTextureSet;
        int occlusionTextureSet;
        int emissiveTextureSet;
        float metallicFactor;
        float roughnessFactor;
        float alphaMask;
        float alphaMaskCutoff;
    } pushConstBlockMaterial;

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
        bool doubleSided;
        Texture2D *metallicRoughnessTexture;
        Texture2D *emissiveTexture;
        Texture2D *occlusionTexture;

        struct Extension {
            Texture2D *specularGlossinessTexture;
            Texture2D *diffuseTexture;
            glm::vec4 diffuseFactor = glm::vec4(1.0f);
            glm::vec3 specularFactor = glm::vec3(0.0f);
        } extension;
    };

    struct TextureIndices{
        int baseColor = -1;
        int normalMap = -1;
    } ;

    struct BoundingBox {
        glm::vec3 min;
        glm::vec3 max;
        bool valid = false;
        BoundingBox();
        BoundingBox(glm::vec3 min, glm::vec3 max);
        BoundingBox getAABB(glm::mat4 m);
    };

    struct Primitive {
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        uint32_t vertexCount = 0;
        Material &material;
        bool hasIndices = false;
        Primitive(uint32_t firstIdx, uint32_t idxCount, uint32_t vtxCount, Material &mat) : firstIndex(firstIdx), indexCount(idxCount), vertexCount(vtxCount), material(mat) {
            hasIndices = indexCount > 0;
        };
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
        } uniformBuffer;
        struct UniformBlock {
            glm::mat4 matrix;
        } uniformBlock;

        Mesh(VulkanDevice *pDevice, glm::mat4 mat1){
            this->device = pDevice;
            this->uniformBlock.matrix = mat1;
            (device->createBuffer(
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    sizeof(uniformBlock),
                    &uniformBuffer.buffer,
                    &uniformBuffer.memory,
                    &uniformBlock));
            (vkMapMemory(device->m_LogicalDevice, uniformBuffer.memory, 0, sizeof(uniformBlock), 0, &uniformBuffer.mapped));
            uniformBuffer.descriptor = { uniformBuffer.buffer, 0, sizeof(uniformBlock) };
        }

        ~Mesh(){
            vkUnmapMemory(device->m_LogicalDevice, uniformBuffer.memory);
            vkFreeMemory(device->m_LogicalDevice, uniformBuffer.memory, nullptr);
            vkDestroyBuffer(device->m_LogicalDevice, uniformBuffer.buffer, nullptr);
        }
    };

    struct LoaderInfo {
        uint32_t* indexBuffer;
        VkRender::Vertex* vertexBuffer;
        size_t indexPos = 0;
        size_t vertexPos = 0;
    };
    struct Node {
        Node *parent;
        uint32_t index;
        std::vector<Node*> children;
        glm::mat4 matrix;
        std::string name;
        Mesh *mesh = nullptr;
        int32_t skinIndex = -1;
        glm::vec3 translation{ 0.0f};
        glm::vec3 scale{ 1.0f };
        glm::quat rotation{};
        glm::mat4 localMatrix();
        glm::mat4 getMatrix();
        void update();
        ~Node() {
            if (mesh)
                delete mesh;
        }
    };

    struct Skin {
        std::string name;
        Node *skeletonRoot = nullptr;
        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<Node*> joints;
    };

    struct Model {
        VulkanDevice *vulkanDevice = nullptr;
        std::string m_FileName;
        explicit Model(VulkanDevice* dev){
            this->vulkanDevice = dev;
        }
        explicit Model(VkRender::RenderUtils* r, VulkanDevice* device = nullptr){
            this->vulkanDevice = device == nullptr ? r->device : device;
            prefilterEnv = r->skybox.prefilterEnv;
            irradianceCube = r->skybox.irradianceCube;
            lutBrdf = r->skybox.lutBrdf;

        }
        ~Model();

        std::vector<Skin*> skins;
        std::vector<std::string> extensions;
        std::vector<Texture2D> textures;
        Texture2D emptyTexture;

        TextureCubeMap *irradianceCube;
        TextureCubeMap *prefilterEnv;
        Texture2D *lutBrdf;

        std::vector<Material> materials;
        std::vector<Texture::TextureSampler> textureSamplers;
        TextureIndices textureIndices;
        bool useCustomTranslation = false;
        glm::vec3 nodeTranslation{};
        glm::vec3 nodeScale = glm::vec3(1.0f, 1.0f, 1.0f);

        bool gltfModelLoaded = false;

        struct Vertices {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory= VK_NULL_HANDLE;
        } vertices{};
        struct Indices {
            int count = 0;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory= VK_NULL_HANDLE;
        } indices{};

        std::vector<Node*> nodes{};
        std::vector<Node*> linearNodes{};

        struct Dimensions {
            glm::vec3 min = glm::vec3(FLT_MAX);
            glm::vec3 max = glm::vec3(-FLT_MAX);
        } dimensions{};

        std::vector<VkDescriptorSet> descriptors;


        void destroy(VkDevice device);
        void loadNode(GLTFModel::Node *parent, const tinygltf::Node &node, uint32_t nodeIndex,
                      const tinygltf::Model &_model, float globalscale, LoaderInfo &loaderInfo);
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



        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptorSetLayoutMaterial = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptorSetLayoutNode = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        struct Pipelines {
            VkPipeline pbr = VK_NULL_HANDLE;
            VkPipeline pbrDoubleSided = VK_NULL_HANDLE;
            VkPipeline pbrAlphaBlend = VK_NULL_HANDLE;
            VkPipeline skybox = VK_NULL_HANDLE;
        } pipelines{};
        VkPipelineLayout pipelineLayout{};
        VkPipeline boundPipeline = VK_NULL_HANDLE;

        void createDescriptors(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo);

        void setupNodeDescriptorSet(Node *node);

        void createPipeline(VkRenderPass renderPass, std::vector<VkPipelineShaderStageCreateInfo> shaderStages);


        void draw(VkCommandBuffer commandBuffer, uint32_t i);
        void drawNode(Node *node, VkCommandBuffer commandBuffer, uint32_t cbIndex, Material::AlphaMode mode);

        void createRenderPipeline(const VkRender::RenderUtils& utils, const std::vector<VkPipelineShaderStageCreateInfo>& shaders, VkCommandPool cmdPool = VK_NULL_HANDLE);


        void translate(const glm::vec3 &translation);
        void scale(const glm::vec3 &scale);

        void createDescriptorsAdditionalBuffers(const std::vector<VkRender::RenderDescriptorBuffersData> &ubo);

        void
        createRenderPipeline(const VkRender::RenderUtils &utils,
                             const std::vector<VkPipelineShaderStageCreateInfo> &shaders,
                             const std::vector<VkRender::RenderDescriptorBuffersData> &buffers, ScriptType flags);

        void createDescriptorSetLayoutAdditionalBuffers();

        void generateBRDFLUT(const std::vector<VkPipelineShaderStageCreateInfo> vector, VkRender::SkyboxTextures *skyboxTextures);
        void setupSkyboxDescriptors(const std::vector<VkRender::UniformBufferSet> &vector, VkRender::SkyboxTextures *skyboxTextures);
        void generateCubemaps(const std::vector<VkPipelineShaderStageCreateInfo> vector,
                              VkRender::SkyboxTextures *skyboxTextures);
        void createSkybox(const std::vector<VkPipelineShaderStageCreateInfo> &envShaders,
                          const std::vector<VkRender::UniformBufferSet> &uboVec,
                          VkRenderPass const *renderPass, VkRender::SkyboxTextures *skyboxTextures);

        void drawSkybox(VkCommandBuffer commandBuffer, uint32_t i);

        void
        getNodeProps(const tinygltf::Node &node, const tinygltf::Model &model, size_t &vertexCount, size_t &indexCount);

        void
        createOpaqueGraphicsPipeline(VkRenderPass const *renderPass, std::vector<VkPipelineShaderStageCreateInfo> shaders);

        void drawNode(Node *node, VkCommandBuffer commandBuffer);
    };



};


#endif //MULTISENSE_GLTFMODEL_H