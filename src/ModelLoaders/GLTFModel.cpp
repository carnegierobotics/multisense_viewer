//
// Created by magnus on 12/10/21.
//

#define TINYGLTF_IMPLEMENTATION

#include <glm/ext.hpp>
#include <stb_image.h>

#include "Viewer/ModelLoaders/GLTFModel.h"
#include "Viewer/Tools/Logger.h"

void GLTFModel::Model::loadFromFile(std::string fileName, VulkanDevice *_device, VkQueue transferQueue, float scale) {
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF gltfContext;
    std::string error;
    std::string warning;

    m_Device = _device;
    m_FileName = fileName;
    Log::Logger::getInstance()->info("Loading glTF file {}", fileName);
    bool binary = false;
    size_t extpos = fileName.rfind('.', fileName.length());
    if (extpos != std::string::npos) {
        binary = (fileName.substr(extpos + 1, fileName.length() - extpos) == "glb");
    }

    bool fileLoaded = binary ? gltfContext.LoadBinaryFromFile(&gltfModel, &error, &warning, fileName.c_str())
                             : gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warning, fileName.c_str());

    if (!fileLoaded){
        std::cerr << warning << std::endl;
        return;
    }
    std::vector<uint32_t> indexBuffer;
    std::vector<Vertex> vertexBuffer;

    const tinygltf::Scene &scene = gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0];
    for (size_t i = 0; i < scene.nodes.size(); i++) {
        const tinygltf::Node node = gltfModel.nodes[scene.nodes[i]];
        loadNode(nullptr, node, scene.nodes[i], gltfModel, indexBuffer, vertexBuffer, scale);
    }

    loadTextureSamplers(gltfModel);
    loadTextures(gltfModel, m_Device, transferQueue);
    loadMaterials(gltfModel);
    extensions = gltfModel.extensionsUsed;

    size_t vertexBufferSize = vertexBuffer.size() * sizeof(Vertex);
    size_t indexBufferSize = indexBuffer.size() * sizeof(uint32_t);
    indices.count = static_cast<uint32_t>(indexBuffer.size());


    assert(vertexBufferSize > 0);

    struct StagingBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    } vertexStaging{}, indexStaging{};

    // Create staging buffers
    // Vertex data
    CHECK_RESULT(m_Device->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize,
            &vertexStaging.buffer,
            &vertexStaging.memory,
            vertexBuffer.data()));
    // Index data
    if (indexBufferSize > 0) {
        CHECK_RESULT(m_Device->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                indexBuffer.data()));
    }

    // Create m_Device local buffers
    // Vertex buffer
    CHECK_RESULT(m_Device->createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vertexBufferSize,
            &vertices.buffer,
            &vertices.memory));
    // Index buffer
    if (indexBufferSize > 0) {
        CHECK_RESULT(m_Device->createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                indexBufferSize,
                &indices.buffer,
                &indices.memory));
    }

    VkBufferCopy copyRegion = {};
    copyRegion.size = vertexBufferSize;
    m_Device->copyVkBuffer(&vertexStaging.buffer, &vertices.buffer, &copyRegion);

    if (indexBufferSize > 0) {
        copyRegion.size = indexBufferSize;
        m_Device->copyVkBuffer(&indexStaging.buffer, &indices.buffer, &copyRegion);
    }


    vkDestroyBuffer(m_Device->m_LogicalDevice, vertexStaging.buffer, nullptr);
    vkFreeMemory(m_Device->m_LogicalDevice, vertexStaging.memory, nullptr);
    if (indexBufferSize > 0) {
        vkDestroyBuffer(m_Device->m_LogicalDevice, indexStaging.buffer, nullptr);
        vkFreeMemory(m_Device->m_LogicalDevice, indexStaging.memory, nullptr);
    }
}


void GLTFModel::Model::translate(const glm::vec3 &translation) {
    nodeTranslation = translation;
    useCustomTranslation = true;
}

void GLTFModel::Model::scale(const glm::vec3 &scale) {
    nodeScale = scale;
}



// TODO: Support multiple children Nodes
void GLTFModel::Model::drawNode(Node *node, VkCommandBuffer commandBuffer) {
    //if (node->m_Mesh)
        for (Primitive primitive: primitives) {
            vkCmdDrawIndexed(commandBuffer, primitive.m_IndexCount, 1, primitive.m_FirstIndex, 0, 0);
        }

    for (auto &child: node->children) {
        drawNode(child, commandBuffer);
    }

}

void GLTFModel::Model::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                            &descriptors[i], 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
    if (indices.buffer != VK_NULL_HANDLE) {
        vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    }
    for (auto &node: nodes) {
        drawNode(node, commandBuffer);
    }
}

/**
 * Function to load each node specificed in glTF m_Format.
 * Recursive call which also loads childnodes and creates and transforms the m_Mesh.
 *
 */
void GLTFModel::Model::loadNode(GLTFModel::Node *parent, const tinygltf::Node &node, uint32_t nodeIndex,
                                const tinygltf::Model &_model, std::vector<uint32_t> &indexBuffer,
                                std::vector<Vertex> &vertexBuffer, float globalscale) {

    auto *newNode = new Node{};
    newNode->index = nodeIndex;
    newNode->parent = parent;
    newNode->name = node.name;
    newNode->skinIndex = node.skin;
    newNode->matrix = glm::mat4(1.0f);

    // Generate local node matrix
    auto translation = glm::vec3(0.0f);
    if (node.translation.size() == 3) {
        translation = glm::make_vec3(node.translation.data());
        newNode->translation = translation;
    }

    if (useCustomTranslation)
        newNode->translation = nodeTranslation;


    if (node.rotation.size() == 4) {
        glm::quat q = glm::make_quat(node.rotation.data());
        newNode->rotation = glm::mat4(q);
    }
    glm::vec3 scale = glm::vec3(1.0f);
    if (node.scale.size() == 3) {
        scale = glm::make_vec3(node.scale.data());
        newNode->scale = scale;
    }
    if (node.matrix.size() == 16) {
        newNode->matrix = glm::make_mat4x4(node.matrix.data());
    };

    // Node with children
    if (node.children.size() > 0) {
        for (size_t i = 0; i < node.children.size(); i++) {
            loadNode(newNode, _model.nodes[node.children[i]], node.children[i], _model, indexBuffer, vertexBuffer,
                     globalscale);
        }
    }

    // Node contains m_Mesh data
    if (node.mesh > -1) {
        const tinygltf::Mesh mesh = _model.meshes[node.mesh];
        for (auto &primitive: mesh.primitives) {
            uint32_t indexStart = static_cast<uint32_t>(indexBuffer.size());
            uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
            uint32_t indexCount = 0;
            glm::vec3 posMin{};
            glm::vec3 posMax{};
            bool hasIndices = primitive.indices > -1;

            // Vertices
            {
                const float *bufferPos = nullptr;
                const float *bufferNormals = nullptr;
                const float *bufferTexCoordSet0 = nullptr;
                const float *bufferTexCoordSet1 = nullptr;

                int posByteStride = 0;
                int normByteStride = 0;
                int uv0ByteStride = 0;
                int uv1ByteStride = 0;

                // Position attribute is required
                assert(primitive.attributes.find("POSITION") != primitive.attributes.end());

                const tinygltf::Accessor &posAccessor = _model.accessors[primitive.attributes.find("POSITION")->second];
                const tinygltf::BufferView &posView = _model.bufferViews[posAccessor.bufferView];
                bufferPos = reinterpret_cast<const float *>(&(_model.buffers[posView.buffer].data[posAccessor.byteOffset + posView.byteOffset]));
                posMin = glm::vec3(posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]);
                posMax = glm::vec3(posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]);
                posByteStride = posAccessor.ByteStride(posView) ? (posAccessor.ByteStride(posView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);

                if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                    const tinygltf::Accessor &normAccessor = _model.accessors[primitive.attributes.find("NORMAL")->second];
                    const tinygltf::BufferView &normView = _model.bufferViews[normAccessor.bufferView];
                    bufferNormals = reinterpret_cast<const float *>(&(_model.buffers[normView.buffer].data[normAccessor.byteOffset + normView.byteOffset]));
                    normByteStride = normAccessor.ByteStride(normView) ? (normAccessor.ByteStride(normView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
                }

                if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                    const tinygltf::Accessor &uvAccessor = _model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
                    const tinygltf::BufferView &uvView = _model.bufferViews[uvAccessor.bufferView];
                    bufferTexCoordSet0 = reinterpret_cast<const float *>(&(_model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
                    uv0ByteStride = uvAccessor.ByteStride(uvView) ? (uvAccessor.ByteStride(uvView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
                }
                if (primitive.attributes.find("TEXCOORD_1") != primitive.attributes.end()) {
                    const tinygltf::Accessor &uvAccessor = _model.accessors[primitive.attributes.find("TEXCOORD_1")->second];
                    const tinygltf::BufferView &uvView = _model.bufferViews[uvAccessor.bufferView];
                    bufferTexCoordSet1 = reinterpret_cast<const float *>(&(_model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));
                    uv1ByteStride = uvAccessor.ByteStride(uvView) ? (uvAccessor.ByteStride(uvView) / sizeof(float)) : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
                }

                for (size_t v = 0; v < posAccessor.count; v++) {
                    Vertex vert{};
                    vert.pos = (glm::vec4(glm::make_vec3(&bufferPos[v * posByteStride]), 1.0f) + glm::vec4( newNode->translation, 1.0f)) * glm::vec4(nodeScale, 1.0f);
                    vert.normal = glm::normalize(glm::vec3(bufferNormals ? glm::make_vec3(&bufferNormals[v * normByteStride]) : glm::vec3(0.0f)));
                    vert.uv0 = bufferTexCoordSet0 ? glm::make_vec2(&bufferTexCoordSet0[v * uv0ByteStride]) : glm::vec3(0.0f);
                    vert.uv1 = bufferTexCoordSet1 ? glm::make_vec2(&bufferTexCoordSet1[v * uv1ByteStride]) : glm::vec3(0.0f);
                    vert.joint0 = glm::vec4(0.0f);
                    vert.weight0 = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                    vertexBuffer.push_back(vert);
                }
            }
            // Indices
            if (hasIndices)
            {
                const tinygltf::Accessor &accessor = _model.accessors[primitive.indices > -1 ? primitive.indices : 0];
                const tinygltf::BufferView &bufferView = _model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer &buffer = _model.buffers[bufferView.buffer];

                indexCount = static_cast<uint32_t>(accessor.count);
                const void *dataPtr = &(buffer.data[accessor.byteOffset + bufferView.byteOffset]);

                switch (accessor.componentType) {
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
                        const uint32_t *buf = static_cast<const uint32_t*>(dataPtr);
                        for (size_t index = 0; index < accessor.count; index++) {
                            indexBuffer.push_back(buf[index] + vertexStart);
                        }
                        break;
                    }
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
                        const uint16_t *buf = static_cast<const uint16_t*>(dataPtr);
                        for (size_t index = 0; index < accessor.count; index++) {
                            indexBuffer.push_back(buf[index] + vertexStart);
                        }
                        break;
                    }
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
                        const uint8_t *buf = static_cast<const uint8_t*>(dataPtr);
                        for (size_t index = 0; index < accessor.count; index++) {
                            indexBuffer.push_back(buf[index] + vertexStart);
                        }
                        break;
                    }
                    default:
                        std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
                        return;
                }
            }
            Primitive newPrimitive(indexStart, indexCount);
            primitives.push_back(newPrimitive);
        }
    }
    if (parent) {
        parent->children.push_back(newNode);
    } else {
        nodes.push_back(newNode);
    }

    linearNodes.push_back(newNode);
}

void GLTFModel::Model::loadTextureSamplers(tinygltf::Model &gltfModel) {
    for (const tinygltf::Sampler &smpl: gltfModel.samplers) {
        Texture::TextureSampler sampler{};
        sampler.minFilter = getVkFilterMode(smpl.minFilter);
        sampler.magFilter = getVkFilterMode(smpl.magFilter);
        sampler.addressModeU = getVkWrapMode(smpl.wrapS);
        sampler.addressModeV = getVkWrapMode(smpl.wrapT);
        sampler.addressModeW = sampler.addressModeV;
        textureSamplers.push_back(sampler);
    }
}

void GLTFModel::Model::loadTextures(tinygltf::Model &gltfModel, VulkanDevice *device, VkQueue transferQueue) {
    for (tinygltf::Texture &tex: gltfModel.textures) {
        tinygltf::Image image = gltfModel.images[tex.source];
        Texture::TextureSampler sampler{};
        if (tex.sampler == -1) {
            // No m_Sampler specified, use a default one
            sampler.magFilter = VK_FILTER_LINEAR;
            sampler.minFilter = VK_FILTER_LINEAR;
            sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        } else {
            sampler = textureSamplers[tex.sampler];
        }
        //Texture2D texture2D(device);
        //texture2D.fromglTfImage(image, sampler, device, transferQueue);
        //textures.push_back(texture2D);
    }


}

void GLTFModel::Model::loadMaterials(tinygltf::Model &gltfModel) {
   /* for (tinygltf::Material &mat: gltfModel.materials) {
        GLTFModel::Material material{};
        if (mat.values.find("baseColorTexture") != mat.values.end()) {
            material.baseColorTexture = &textures[mat.values["baseColorTexture"].TextureIndex()];
            material.texCoordSets.baseColor = mat.values["baseColorTexture"].TextureTexCoord();
            textureIndices.baseColor = 0;

        }
        if (mat.values.find("baseColorFactor") != mat.values.end()) {
            material.baseColorFactor = glm::make_vec4(mat.values["baseColorFactor"].ColorFactor().data());
        }

        if (mat.additionalValues.find("normalTexture") != mat.additionalValues.end()) {
            textureIndices.normalMap = mat.additionalValues["normalTexture"].TextureIndex();
            material.normalTexture = &textures[mat.additionalValues["normalTexture"].TextureIndex()];
            material.texCoordSets.normal = mat.additionalValues["normalTexture"].TextureTexCoord();
        }

        materials.push_back(material);
    }
    */
}

VkSamplerAddressMode GLTFModel::Model::getVkWrapMode(int32_t wrapMode) {
    switch (wrapMode) {
        case 10497:
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case 33071:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case 33648:
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    }
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
}

VkFilter GLTFModel::Model::getVkFilterMode(int32_t filterMode) {
    switch (filterMode) {
        case 9728:
            return VK_FILTER_NEAREST;
        case 9729:
            return VK_FILTER_LINEAR;
        case 9984:
            return VK_FILTER_NEAREST;
        case 9985:
            return VK_FILTER_NEAREST;
        case 9986:
            return VK_FILTER_LINEAR;
        case 9987:
            return VK_FILTER_LINEAR;
        default:
            std::cerr << "Sampler filter defaulted to VK_FILTER_LINEAR" << std::endl;
            return VK_FILTER_LINEAR;
    }
}

/**
 * Function to set texture other than the specified in embedded glTF file.
 * @param fileName Name of texture. Requires full path
 */
void GLTFModel::Model::setTexture(std::basic_string<char, std::char_traits<char>, std::allocator<char>> fileName) {
    // Create texture m_Image

    int texWidth = 0, texHeight = 0, texChannels = 0;
    stbi_uc *pixels = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth * texHeight * 4);;
    if (!pixels) {
        throw std::runtime_error("failed to load texture m_Image!");
    }

    Texture2D texture(m_Device);
    texture.fromBuffer(pixels, imageSize, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, m_Device, m_Device->m_TransferQueue);
    textureIndices.baseColor = 0;
    textures.push_back(texture);

    Texture::TextureSampler sampler{};
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    textureSamplers.push_back(sampler);


}

// TODO USE ENUMS TO SET COLOR OR NORMAL INSTEAD OF SEPARATE ALMOST INDENTICAL FUNCTIONS
/**
 * Function to set normal texture other than the specified in embedded glTF file.
 * @param fileName Name of texture. Requires full path
 */
void GLTFModel::Model::setNormalMap(std::basic_string<char, std::char_traits<char>, std::allocator<char>> fileName) {

    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = (VkDeviceSize) texWidth * texHeight * 4;
    if (!pixels) {
        throw std::runtime_error("failed to load texture m_Image!");
    }

    Texture2D texture(m_Device);
    texture.fromBuffer(pixels, imageSize, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, m_Device, m_Device->m_TransferQueue);
    textureIndices.normalMap = 1;
    textures.push_back(texture);

    Texture::TextureSampler sampler{};
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    textureSamplers.push_back(sampler);

}


GLTFModel::GLTFModel() {
}


GLTFModel::Primitive::Primitive(uint32_t _firstIndex, uint32_t indexCount) {
    this->m_FirstIndex = _firstIndex;
    this->m_IndexCount = indexCount;
}

void GLTFModel::Model::createDescriptorSetLayout() {
    // TODO BETTER SELECTION PROCESS
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
    };


    if (textureIndices.baseColor != -1) {
        setLayoutBindings.push_back(
                {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
    }

    if (textureIndices.normalMap != -1) {
        setLayoutBindings.push_back(
                {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});

    }


    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
    descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
    descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
    CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr, &descriptorSetLayout))
}

void GLTFModel::Model::createDescriptors(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo) {
    descriptors.resize(count);
    // Check for how many m_Image descriptors
    /**
     * Create Descriptor Pool
     */
    uint32_t uniformDescriptorCount = (3 * count + (uint32_t)nodes.size());
    uint32_t imageDescriptorSamplerCount = (3 * count * 3);
    std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         uniformDescriptorCount},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageDescriptorSamplerCount},

    };
    VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes,
        static_cast<uint32_t>(count + nodes.size()));
    CHECK_RESULT(vkCreateDescriptorPool(vulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr, &descriptorPool));
    /**
     * Create Descriptor Sets
     */
    for (size_t i = 0; i < descriptors.size(); i++) {

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayout;
        descriptorSetAllocInfo.descriptorSetCount = 1;
        CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo, &descriptors[i]));

        std::vector<VkWriteDescriptorSet> writeDescriptorSets(3);

        writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[0].descriptorCount = 1;
        writeDescriptorSets[0].dstSet = descriptors[i];
        writeDescriptorSets[0].dstBinding = 0;
        writeDescriptorSets[0].pBufferInfo = &ubo[i].bufferOne.m_DescriptorBufferInfo;

        writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[1].descriptorCount = 1;
        writeDescriptorSets[1].dstSet = descriptors[i];
        writeDescriptorSets[1].dstBinding = 1;
        writeDescriptorSets[1].pBufferInfo = &ubo[i].bufferTwo.m_DescriptorBufferInfo;

        writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[2].descriptorCount = 1;
        writeDescriptorSets[2].dstSet = descriptors[i];
        writeDescriptorSets[2].dstBinding = 2;
        writeDescriptorSets[2].pBufferInfo = &ubo[i].bufferThree.m_DescriptorBufferInfo;

        vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                               writeDescriptorSets.data(), 0, NULL);
    }


    // Model node (matrices)
    {
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
        descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
        descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
        CHECK_RESULT(
                vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr,
                                            &descriptorSetLayoutNode));

        // Per-Node m_Descriptor set
        for (auto &node: nodes) {
            setupNodeDescriptorSet(node);
        }
    }

}

void GLTFModel::Model::createDescriptorSetLayoutAdditionalBuffers() {
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
    descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
    descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
    CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr, &descriptorSetLayout))
}

void GLTFModel::Model::createDescriptorsAdditionalBuffers(const std::vector<VkRender::RenderDescriptorBuffersData> &ubo) {
    descriptors.resize(ubo.size());
    // Check for how many m_Image descriptors
    /**
     * Create Descriptor Pool
     */
    uint32_t uniformDescriptorCount = (2 * ubo.size() + (uint32_t)nodes.size());
    std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         uniformDescriptorCount}
    };
    VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes,
                                                                                   static_cast<uint32_t>(ubo.size() + nodes.size()));
    CHECK_RESULT(vkCreateDescriptorPool(vulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr, &descriptorPool));
    /**
     * Create Descriptor Sets
     */
    for (size_t i = 0; i < descriptors.size(); i++) {

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayout;
        descriptorSetAllocInfo.descriptorSetCount = 1;
        CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo, &descriptors[i]));

        std::vector<VkWriteDescriptorSet> writeDescriptorSets(2);

        writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[0].descriptorCount = 1;
        writeDescriptorSets[0].dstSet = descriptors[i];
        writeDescriptorSets[0].dstBinding = 0;
        writeDescriptorSets[0].pBufferInfo = &ubo[i].mvp.m_DescriptorBufferInfo;

        writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[1].descriptorCount = 1;
        writeDescriptorSets[1].dstSet = descriptors[i];
        writeDescriptorSets[1].dstBinding = 1;
        writeDescriptorSets[1].pBufferInfo = &ubo[i].light.m_DescriptorBufferInfo;

        vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                               writeDescriptorSets.data(), 0, NULL);
    }


    // Model node (matrices)
    {
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
        descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
        descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
        CHECK_RESULT(
                vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr,
                                            &descriptorSetLayoutNode));

        // Per-Node m_Descriptor set
        for (auto &node: nodes) {
            setupNodeDescriptorSet(node);
        }
    }

}

void GLTFModel::Model::setupNodeDescriptorSet(GLTFModel::Node *node) {
    if (node->mesh) {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayoutNode;
        descriptorSetAllocInfo.descriptorSetCount = 1;
        CHECK_RESULT(
                vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                         &node->mesh->uniformBuffer.descriptorSet));

        VkWriteDescriptorSet writeDescriptorSet{};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.dstSet = node->mesh->uniformBuffer.descriptorSet;
        writeDescriptorSet.dstBinding = 0;
        writeDescriptorSet.pBufferInfo = &node->mesh->uniformBuffer.descriptor;

        vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, 1, &writeDescriptorSet, 0, nullptr);
    }
    for (auto &child: node->children) {
        setupNodeDescriptorSet(child);
    }
}


void GLTFModel::Model::createPipeline(VkRenderPass renderPass, std::vector<VkPipelineShaderStageCreateInfo> shaderStages) {

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = Populate::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);

    VkPipelineRasterizationStateCreateInfo rasterizationStateCI = Populate::pipelineRasterizationStateCreateInfo(
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
    rasterizationStateCI.lineWidth = 1.0f;

    VkPipelineColorBlendAttachmentState blendAttachmentState = Populate::pipelineColorBlendAttachmentState(
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            VK_FALSE);

    VkPipelineColorBlendStateCreateInfo colorBlendStateCI = Populate::pipelineColorBlendStateCreateInfo(1,
                                                                                                        &blendAttachmentState);

    VkPipelineDepthStencilStateCreateInfo depthStencilStateCI =
            Populate::pipelineDepthStencilStateCreateInfo(VK_TRUE,
                                                          VK_TRUE,
                                                          VK_COMPARE_OP_LESS);
    depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
    depthStencilStateCI.stencilTestEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportStateCI{};
    viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateCI.viewportCount = 1;
    viewportStateCI.scissorCount = 1;

    VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
    multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;


    std::vector<VkDynamicState> dynamicStateEnables = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicStateCI{};
    dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
    dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());


    // Pipeline layout
    const std::vector<VkDescriptorSetLayout> setLayouts = {
            descriptorSetLayout, descriptorSetLayoutNode
    };
    VkPipelineLayoutCreateInfo pipelineLayoutCI{};
    pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCI.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutCI.pSetLayouts = setLayouts.data();
    pipelineLayoutCI.pushConstantRangeCount = 0;
    pipelineLayoutCI.pPushConstantRanges = nullptr;
    CHECK_RESULT(vkCreatePipelineLayout(vulkanDevice->m_LogicalDevice, &pipelineLayoutCI, nullptr, &pipelineLayout));


    // Vertex bindings an attributes
    VkVertexInputBindingDescription vertexInputBinding = {0, sizeof(GLTFModel::Model::Vertex),
                                                          VK_VERTEX_INPUT_RATE_VERTEX};
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3},
            {2, 0, VK_FORMAT_R32G32_SFLOAT,    sizeof(float) * 6},
            {3, 0, VK_FORMAT_R32G32_SFLOAT,    sizeof(float) * 8},
    };
    VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
    vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputStateCI.vertexBindingDescriptionCount = 1;
    vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
    vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
    vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

    // Pipelines
    VkGraphicsPipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCI.layout = pipelineLayout;
    pipelineCI.renderPass = renderPass;
    pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
    pipelineCI.pVertexInputState = &vertexInputStateCI;
    pipelineCI.pRasterizationState = &rasterizationStateCI;
    pipelineCI.pColorBlendState = &colorBlendStateCI;
    pipelineCI.pMultisampleState = &multisampleStateCI;
    pipelineCI.pViewportState = &viewportStateCI;
    pipelineCI.pDepthStencilState = &depthStencilStateCI;
    pipelineCI.pDynamicState = &dynamicStateCI;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();
    multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;


    CHECK_RESULT(vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr, &pipeline));
}

void GLTFModel::Model::createRenderPipeline(const VkRender::RenderUtils& utils, const std::vector<VkPipelineShaderStageCreateInfo>& shaders) {
    this->vulkanDevice = utils.device;
    createDescriptorSetLayout();
    createDescriptors(utils.UBCount, utils.uniformBuffers);
    createPipeline(*utils.renderPass, shaders);
}

void GLTFModel::Model::createRenderPipeline(const VkRender::RenderUtils& utils, const std::vector<VkPipelineShaderStageCreateInfo>& shaders, const std::vector<VkRender::RenderDescriptorBuffersData>& buffers, ScriptType flags) {
    this->vulkanDevice = utils.device;
    if (flags == AR_SCRIPT_TYPE_ADDITIONAL_BUFFERS){
        createDescriptorSetLayoutAdditionalBuffers();
        createDescriptorsAdditionalBuffers(buffers);
        createPipeline(*utils.renderPass, shaders);
    }

}

GLTFModel::Model::~Model() {
    {
        vkFreeMemory(m_Device->m_LogicalDevice, vertices.memory, nullptr);
        vkFreeMemory(m_Device->m_LogicalDevice, indices.memory, nullptr);
        vkDestroyBuffer(m_Device->m_LogicalDevice, vertices.buffer, nullptr);
        vkDestroyBuffer(m_Device->m_LogicalDevice, indices.buffer, nullptr);

        vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorSetLayoutNode, nullptr);
        vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, descriptorPool, nullptr);
        vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, pipelineLayout, nullptr);
        vkDestroyPipeline(vulkanDevice->m_LogicalDevice, pipeline, nullptr);

        for(auto* node : linearNodes){
            delete node;
        }

        Log::Logger::getInstance()->info("Destroying model {}", m_FileName);

    }
}

