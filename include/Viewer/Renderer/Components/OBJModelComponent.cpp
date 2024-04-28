//
// Created by magnus on 4/11/24.
//

#include "Viewer/Renderer/Components/OBJModelComponent.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/Tools/Utils.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#define TINYOBJLOADER_USE_MAPBOX_EARCUT

#include <tiny_obj_loader.h>
#include <stb_image.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
namespace std {
    template<> struct hash<VkRender::Vertex> {
        size_t operator()(VkRender::Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                     (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.uv0) << 1);
        }
    };
};

namespace VkRender {



    void OBJModelComponent::loadModel(std::filesystem::path modelPath) {


        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files

        tinyobj::ObjReader reader;


        if (!reader.ParseFromFile(modelPath, reader_config)) {
            if (!reader.Error().empty()) {
                std::cerr << "TinyObjReader: " << reader.Error();
                Log::Logger::getInstance()->error("Failed to load .OBJ file {}", modelPath.c_str());
            }
            return;
        }


        if (!reader.Warning().empty()) {
            Log::Logger::getInstance()->warning(".OBJ file empty {}", modelPath.c_str());
        }

        auto &attrib = reader.GetAttrib();
        auto &shapes = reader.GetShapes();
        auto &materials = reader.GetMaterials();0;


// Pre-allocate memory for vertices and indices vectors
        size_t estimatedVertexCount = attrib.vertices.size() / 3;
        size_t estimatedIndexCount = 0;
        for (const auto& shape : shapes) {
            estimatedIndexCount += shape.mesh.indices.size();
        }

        std::unordered_map<VkRender::Vertex, uint32_t> uniqueVertices{};
        uniqueVertices.reserve(estimatedVertexCount);  // Reserve space for unique vertices

        std::vector<VkRender::Vertex> verts;
        std::vector<uint32_t> idx;
        verts.reserve(estimatedVertexCount);  // Reserve space to avoid resizing
        idx.reserve(estimatedIndexCount);     // Reserve space to avoid resizing

        // Using range-based loop to process vertices and indices
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};
                vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.uv0 = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                auto [it, inserted] = uniqueVertices.try_emplace(vertex, verts.size());
                if (inserted) {
                    verts.push_back(vertex);
                }

                idx.push_back(it->second);
            }
        }
        size_t vertexBufferSize = verts.size() * sizeof(VkRender::Vertex);
        size_t indexBufferSize = idx.size() * sizeof(uint32_t);
        indices.indexCount = idx.size();

        assert(vertexBufferSize > 0);

        struct StagingBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
        } vertexStaging, indexStaging{};

        // Create staging buffers
        // Vertex data
        CHECK_RESULT(device->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertexBufferSize,
                &vertexStaging.buffer,
                &vertexStaging.memory,
                verts.data()));
        // Index data
        if (indexBufferSize > 0) {
            CHECK_RESULT(device->createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    indexBufferSize,
                    &indexStaging.buffer,
                    &indexStaging.memory,
                    idx.data()));
        }

        // Create device local buffers
        // Vertex buffer
        CHECK_RESULT(device->createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                vertexBufferSize,
                &vertices.buffer,
                &vertices.memory));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(device->createBuffer(
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    indexBufferSize,
                    &indices.buffer,
                    &indices.memory));
        }

        // Copy from staging buffers
        VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        VkBufferCopy copyRegion = {};

        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, vertices.buffer, 1, &copyRegion);

        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, indices.buffer, 1, &copyRegion);
        }

        device->flushCommandBuffer(copyCmd, device->m_TransferQueue, true);

        vkDestroyBuffer(device->m_LogicalDevice, vertexStaging.buffer, nullptr);
        vkFreeMemory(device->m_LogicalDevice, vertexStaging.memory, nullptr);
        if (indexBufferSize > 0) {
            vkDestroyBuffer(device->m_LogicalDevice, indexStaging.buffer, nullptr);
            vkFreeMemory(device->m_LogicalDevice, indexStaging.memory, nullptr);
        }
    }

    void OBJModelComponent::loadTexture(std::filesystem::path texturePath) {
        int texWidth = 0, texHeight = 0, texChannels = 0;
        auto path = texturePath.replace_extension(".png");
        stbi_uc *pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!pixels) {
            Log::Logger::getInstance()->error("Failed to load texture: {}", texturePath.c_str());
            pixels = stbi_load((Utils::getTexturePath() / "moon.png").c_str(), &texWidth, &texHeight, &texChannels,
                               STBI_rgb_alpha);

        }
        VkDeviceSize size = 4 * texHeight * texWidth;
        objTexture.fromBuffer(pixels, size, VK_FORMAT_R8G8B8A8_SRGB,
                              texWidth, texHeight, device,
                              device->m_TransferQueue);

        stbi_image_free(pixels);

    }

    void OBJModelComponent::draw(CommandBuffer *commandBuffer, uint32_t cbIndex) {
        VkDeviceSize offsets[1] = {0};

        vkCmdBindVertexBuffers(commandBuffer->buffers[cbIndex], 0, 1, &vertices.buffer, offsets);
        if (indices.buffer != VK_NULL_HANDLE) {
            vkCmdBindIndexBuffer(commandBuffer->buffers[cbIndex], indices.buffer, 0,
                                 VK_INDEX_TYPE_UINT32);
        }

        if (indices.buffer != VK_NULL_HANDLE) {
            vkCmdDrawIndexed(commandBuffer->buffers[cbIndex], indices.indexCount, 1,
                             0, 0, 0);
        } else {
            vkCmdDraw(commandBuffer->buffers[cbIndex], vertices.vertexCount, 1, 0, 0);
        }
    }
};