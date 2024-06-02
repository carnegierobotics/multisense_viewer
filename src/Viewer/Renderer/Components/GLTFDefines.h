//
// Created by magnus on 4/12/24.
//

#ifndef MULTISENSE_VIEWER_GLTFDEFINES_H
#define MULTISENSE_VIEWER_GLTFDEFINES_H

#include "Viewer/Core/Texture.h"
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/fwd.hpp>

#include "Viewer/Core/RenderDefinitions.h"

#define MAX_NUM_JOINTS 128u

namespace VkRender {
    struct Node;
    struct BoundingBox {
        glm::vec3 min{};
        glm::vec3 max{};
        bool valid = false;

        BoundingBox();

        BoundingBox(glm::vec3 min, glm::vec3 max);

        BoundingBox getAABB(glm::mat4 m) const;
    };


    struct Material {
        enum AlphaMode {
            ALPHAMODE_OPAQUE, ALPHAMODE_MASK, ALPHAMODE_BLEND
        };
        AlphaMode alphaMode = ALPHAMODE_OPAQUE;
        float alphaCutoff = 1.0f;
        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        glm::vec4 emissiveFactor = glm::vec4(0.0f);
        Texture2D *baseColorTexture;
        Texture2D *metallicRoughnessTexture;
        Texture2D *normalTexture;
        Texture2D *occlusionTexture;
        Texture2D *emissiveTexture;
        bool doubleSided = false;
        struct TexCoordSets {
            uint8_t baseColor = 0;
            uint8_t metallicRoughness = 0;
            uint8_t specularGlossiness = 0;
            uint8_t normal = 0;
            uint8_t occlusion = 0;
            uint8_t emissive = 0;
        } texCoordSets;
        struct Extension {
            Texture2D *specularGlossinessTexture;
            Texture2D *diffuseTexture;
            glm::vec4 diffuseFactor = glm::vec4(1.0f);
            glm::vec3 specularFactor = glm::vec3(0.0f);
        } extension;
        struct PbrWorkflows {
            bool metallicRoughness = true;
            bool specularGlossiness = false;
        } pbrWorkflows;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        int index = 0;
        bool unlit = false;
        float emissiveStrength = 1.0f;
    };

    struct Primitive {
        uint32_t firstIndex;
        uint32_t indexCount;
        uint32_t vertexCount;
        Material &material;
        bool hasIndices;
        BoundingBox bb;

        Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, Material &material);

        void setBoundingBox(glm::vec3 min, glm::vec3 max);
    };

    struct Mesh {
        VulkanDevice *m_device;
        std::vector<Primitive *> primitives;
        BoundingBox bb;
        BoundingBox aabb;
        struct UniformBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
            VkDescriptorBufferInfo descriptor;
            VkDescriptorSet descriptorSet;
            void *mapped;
        } uniformBuffer;
        struct UniformBlock {
            glm::mat4 matrix;
            glm::mat4 jointMatrix[MAX_NUM_JOINTS]{};
            float jointcount{0};
        } uniformBlock;

        Mesh(VulkanDevice *device, glm::mat4 matrix);

        ~Mesh();

        void setBoundingBox(glm::vec3 min, glm::vec3 max);
    };

    struct Skin {
        std::string name;
        Node *skeletonRoot = nullptr;
        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<Node *> joints;
    };

    struct Node {
        Node *parent;
        uint32_t index;
        std::vector<Node *> children;
        glm::mat4 matrix;
        std::string name;
        Mesh *mesh;
        Skin *skin;
        int32_t skinIndex = -1;
        glm::vec3 translation{};
        glm::vec3 scale{1.0f};
        glm::quat rotation{};
        BoundingBox bvh;
        BoundingBox aabb;

        glm::mat4 localMatrix();

        glm::mat4 getMatrix();

        void update();

        ~Node();
    };

    struct AnimationChannel {
        enum PathType {
            TRANSLATION, ROTATION, SCALE
        };
        PathType path;
        Node *node;
        uint32_t samplerIndex;
    };

    struct AnimationSampler {
        enum InterpolationType {
            LINEAR, STEP, CUBICSPLINE
        };
        InterpolationType interpolation;
        std::vector<float> inputs;
        std::vector<glm::vec4> outputsVec4;
    };

    struct Animation {
        std::string name;
        std::vector<AnimationSampler> samplers;
        std::vector<AnimationChannel> channels;
        float start = std::numeric_limits<float>::max();
        float end = std::numeric_limits<float>::min();
    };


    struct Model {

        Model(std::filesystem::path modelPath, VulkanDevice *_device) {
            m_device = _device;
            loadFromFile(modelPath.string());
        }

        ~Model(){
            destroy(m_device->m_LogicalDevice);
        }

        VulkanDevice *m_device;

        struct Vertices {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory;
        } vertices;
        struct Indices {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory;
        } indices;

        glm::mat4 aabb;

        std::vector<Node *> nodes;
        std::vector<Node *> linearNodes;

        std::vector<Skin *> skins;

        std::vector<Texture2D> textures;
        std::vector<Texture::TextureSampler> textureSamplers;
        std::vector<Material> materials;
        std::vector<Animation> animations;
        std::vector<std::string> extensions;

        struct Dimensions {
            glm::vec3 min = glm::vec3(FLT_MAX);
            glm::vec3 max = glm::vec3(-FLT_MAX);
        } dimensions;

        struct LoaderInfo {
            uint32_t *indexBuffer;
            VkRender::Vertex *vertexBuffer;
            size_t indexPos = 0;
            size_t vertexPos = 0;
        };


        void destroy(VkDevice device);

        void loadNode(Node *parent, const tinygltf::Node &node, uint32_t nodeIndex, const tinygltf::Model &model,
                      Model::LoaderInfo &loaderInfo, float globalscale);

        void getNodeProps(const tinygltf::Node &node, const tinygltf::Model &model, size_t &vertexCount,
                          size_t &indexCount);

        void loadSkins(tinygltf::Model &gltfModel);

        void loadTextures(tinygltf::Model &gltfModel, VulkanDevice *dev, VkQueue transferQueue);

        VkSamplerAddressMode getVkWrapMode(int32_t wrapMode);

        VkFilter getVkFilterMode(int32_t filterMode);

        void loadTextureSamplers(tinygltf::Model &gltfModel);

        void loadMaterials(tinygltf::Model &gltfModel);

        void loadAnimations(tinygltf::Model &gltfModel);

        void loadFromFile(std::string filename, float scale = 1.0f);

        void drawNode(Node *node, VkCommandBuffer commandBuffer);

        void draw(VkCommandBuffer commandBuffer);

        void calculateBoundingBox(Node *node, Node *parent);

        void getSceneDimensions();

        void updateAnimation(uint32_t index, float time);

        Node *findNode(Node *parent, uint32_t index);

        Node *nodeFromIndex(uint32_t index);
    };
}

#endif //MULTISENSE_VIEWER_GLTFDEFINES_H
