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
        auto &materials = reader.GetMaterials();

        // Loop over shapes
        size_t totalVertexCount = attrib.vertices.size() / 3; // Each vertex has three components (x, y, z)
        size_t totalIndexCount = 0;

        for (const auto & shape : shapes) {
            for (unsigned int num_face_vertice : shape.mesh.num_face_vertices) {
                auto fv = size_t(num_face_vertice);
                totalIndexCount += fv; // Each vertex in the face adds an index
            }
        }

        std::vector<VkRender::Vertex> vertexBuffer(totalVertexCount * 3);
        std::vector<uint32_t> indexBuffer(totalIndexCount);

        vertices.vertexCount = totalVertexCount;
        indices.indexCount = totalIndexCount;


        size_t indexIndex = 0;
        for (const auto &shape : shapes) {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                auto fv = size_t(shape.mesh.num_face_vertices[f]);

                // Loop over vertices in the face.
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                    // Ensure the index points to a valid position in the vertex array.
                    if (idx.vertex_index >= 0 && idx.vertex_index < totalVertexCount) {
                        indexBuffer[indexIndex++] = idx.vertex_index; // Store the vertex index directly

                        // Access and store vertex position
                        tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                        tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                        tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                        vertexBuffer[idx.vertex_index].pos = glm::vec3(vx, vy, vz);

                        // Similarly handle normals and texture coordinates if available
                        if (idx.normal_index >= 0) {
                            tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                            tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                            tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                            vertexBuffer[idx.vertex_index].normal = glm::vec3(nx, ny, nz);
                        }

                        if (idx.texcoord_index >= 0) {
                            tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                            tinyobj::real_t ty = 1.0f - attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                            vertexBuffer[idx.vertex_index].uv0 = glm::vec2(tx, ty);
                        }
                    }
                }
                index_offset += fv;
            }
        }


        size_t vertexBufferSize = totalVertexCount * sizeof(VkRender::Vertex);
        size_t indexBufferSize = totalIndexCount * sizeof(uint32_t);

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
                vertexBuffer.data()));
        // Index data
        if (indexBufferSize > 0) {
            CHECK_RESULT(device->createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    indexBufferSize,
                    &indexStaging.buffer,
                    &indexStaging.memory,
                    indexBuffer.data()));
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
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!pixels){
            Log::Logger::getInstance()->error("Failed to load texture: {}", texturePath.c_str());
            pixels = stbi_load((Utils::getTexturePath() / "moon.png").c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

        }
        VkDeviceSize size = 4 * texHeight * texWidth;
        objTexture.fromBuffer(pixels, size, VK_FORMAT_R8G8B8A8_SRGB,
                              texWidth, texHeight, device,
                              device->m_TransferQueue);

        stbi_image_free(pixels);

    }

    void OBJModelComponent::draw(CommandBuffer* commandBuffer, uint32_t cbIndex){
        VkDeviceSize offsets[1] = { 0 };

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