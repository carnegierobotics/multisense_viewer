//
// Created by magnus on 11/27/24.
//

#include "VulkanMeshResourceManager.h"

#include "Viewer/Application/Application.h"

namespace VkRender {
    std::shared_ptr<MeshInstance> MeshResourceManager::getMeshInstance(
        const std::string& identifier,
        const std::shared_ptr<MeshData>& meshData,
        MeshDataType meshType) {
        std::lock_guard lock(cacheMutex);

        auto it = meshInstanceCache.find(identifier);
        if (it != meshInstanceCache.end()) {
            if (meshData->isDirty) {
                updateMeshInstance(identifier, meshData);
            }
            return it->second;
        }

        // Create new MeshInstance
        auto meshInstance = createMeshInstance(meshData, meshType);
        if (meshInstance) {
            meshInstanceCache[identifier] = meshInstance;
        }

        return meshInstance;
    }

    void MeshResourceManager::updateMeshInstance(
        const std::string& identifier,
        const std::shared_ptr<MeshData>& meshData) {
        auto it = meshInstanceCache.find(identifier);
        if (it != meshInstanceCache.end() && meshData->isDirty) {
            auto meshInstance = it->second;

            VkDeviceSize vertexBufferSize = meshData->vertices.size() * sizeof(Vertex);
            VkDeviceSize indexBufferSize = meshData->indices.size() * sizeof(uint32_t);

            if (vertexBufferSize == 0)
                return;

            if (meshData->isDynamic) {
                void* data;
                vkMapMemory(m_context->vkDevice().m_LogicalDevice, meshInstance->vertexBuffer->m_memory, 0,
                            vertexBufferSize, 0, &data);
                memcpy(data, meshData->vertices.data(), static_cast<size_t>(vertexBufferSize));
                vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, meshInstance->vertexBuffer->m_memory);

                if (indexBufferSize > 0) {
                    vkMapMemory(m_context->vkDevice().m_LogicalDevice, meshInstance->indexBuffer->m_memory, 0,
                                indexBufferSize, 0, &data);
                    memcpy(data, meshData->indices.data(), static_cast<size_t>(indexBufferSize));
                    vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, meshInstance->indexBuffer->m_memory);
                }
            }
            else {
                // For static meshes, use staging buffers to update device local memory
                struct StagingBuffer {
                    VkBuffer buffer;
                    VkDeviceMemory memory;
                } vertexStaging{}, indexStaging{};

                CHECK_RESULT(m_context->vkDevice().createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    vertexBufferSize,
                    &vertexStaging.buffer,
                    &vertexStaging.memory,
                    meshData->vertices.data()));

                if (indexBufferSize > 0) {
                    CHECK_RESULT(m_context->vkDevice().createBuffer(
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        indexBufferSize,
                        &indexStaging.buffer,
                        &indexStaging.memory,
                        meshData->indices.data()));
                }

                // Copy data from staging buffers to device local buffers
                VkCommandBuffer copyCmd = m_context->vkDevice().createCommandBuffer(
                    VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
                VkBufferCopy copyRegion = {};
                copyRegion.size = vertexBufferSize;
                vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, meshInstance->vertexBuffer->m_buffer, 1, &copyRegion);
                if (indexBufferSize > 0) {
                    copyRegion.size = indexBufferSize;
                    vkCmdCopyBuffer(copyCmd, indexStaging.buffer, meshInstance->indexBuffer->m_buffer, 1, &copyRegion);
                }
                m_context->vkDevice().flushCommandBuffer(copyCmd, m_context->vkDevice().m_TransferQueue, true);

                // Clean up staging buffers
                vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, vertexStaging.buffer, nullptr);
                vkFreeMemory(m_context->vkDevice().m_LogicalDevice, vertexStaging.memory, nullptr);
                if (indexBufferSize > 0) {
                    vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, indexStaging.buffer, nullptr);
                    vkFreeMemory(m_context->vkDevice().m_LogicalDevice, indexStaging.memory, nullptr);
                }
            }

            // Reset the dirty flag
            meshData->isDirty = false;
        }
    }

    void MeshResourceManager::clearCache() {
        std::lock_guard<std::mutex> lock(cacheMutex);
        for (auto& pair : meshInstanceCache) {
            // Resources will be cleaned up by MeshInstance destructors
            pair.second.reset();
        }
        meshInstanceCache.clear();
    }

    void MeshResourceManager::removeMeshInstance(const std::string& identifier) {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = meshInstanceCache.find(identifier);
        if (it != meshInstanceCache.end()) {
            it->second.reset();
            meshInstanceCache.erase(it);
        }
    }

    std::shared_ptr<MeshInstance> MeshResourceManager::createMeshInstance(
        const std::shared_ptr<MeshData>& meshData,
        MeshDataType meshType) {
        auto meshInstance = std::make_shared<MeshInstance>();


        meshInstance->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        meshInstance->vertexCount = static_cast<uint32_t>(meshData->vertices.size());
        meshInstance->indexCount = static_cast<uint32_t>(meshData->indices.size());
        VkDeviceSize vertexBufferSize = meshInstance->vertexCount * sizeof(Vertex);
        VkDeviceSize indexBufferSize = meshInstance->indexCount * sizeof(uint32_t);
        meshInstance->m_type = meshType;

        meshInstance->usesVertexBuffers = meshData->isDynamic == false;
        meshInstance->SSBO = meshData->isDynamic;
        if (meshInstance->SSBO) {
            meshInstance->drawCount =  meshInstance->indexCount; // TODO Is not compatible if we use SSBO's with no. vertices as draw count
        }
        if (vertexBufferSize == 0)
            return nullptr;

        // Decide on memory properties
        VkMemoryPropertyFlags memoryProperties;
        VkBufferUsageFlags usageFlags;

        if (meshData->isDynamic) {
            // For dynamic meshes, use host-visible memory
            memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            usageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        }
        else {
            // For static meshes, use device-local memory
            memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            usageFlags = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        }

        // Create vertex buffer
        CHECK_RESULT(m_context->vkDevice().createBuffer(
            usageFlags,
            memoryProperties,
            meshInstance->vertexBuffer,
            vertexBufferSize,
            nullptr,
            "MeshResourceManager:VertexBuffer",
            m_context->getDebugUtilsObjectNameFunction()));

        // Create index buffer if necessary
        if (indexBufferSize > 0) {
            if (meshData->isDynamic) {
                usageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            }
            else {
                usageFlags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            }
            CHECK_RESULT(m_context->vkDevice().createBuffer(
                usageFlags,
                memoryProperties,
                meshInstance->indexBuffer,
                indexBufferSize,
                nullptr,
                "MeshResourceManager:IndexBuffer",
                m_context->getDebugUtilsObjectNameFunction()));
        }

        // Upload data to buffers
        if (meshData->isDynamic) {
            // Map memory and copy data directly
            void* data;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice, meshInstance->vertexBuffer->m_memory, 0,
                        vertexBufferSize, 0, &data);
            memcpy(data, meshData->vertices.data(), static_cast<size_t>(vertexBufferSize));
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, meshInstance->vertexBuffer->m_memory);

            if (indexBufferSize > 0) {
                vkMapMemory(m_context->vkDevice().m_LogicalDevice, meshInstance->indexBuffer->m_memory, 0,
                            indexBufferSize, 0, &data);
                memcpy(data, meshData->indices.data(), static_cast<size_t>(indexBufferSize));
                vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, meshInstance->indexBuffer->m_memory);
            }
        }
        else {
            // Use staging buffers for static meshes
            // Create staging buffers
            struct StagingBuffer {
                VkBuffer buffer;
                VkDeviceMemory memory;
            } vertexStaging{}, indexStaging{};

            CHECK_RESULT(m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertexBufferSize,
                &vertexStaging.buffer,
                &vertexStaging.memory,
                meshData->vertices.data()));

            if (indexBufferSize > 0) {
                CHECK_RESULT(m_context->vkDevice().createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    indexBufferSize,
                    &indexStaging.buffer,
                    &indexStaging.memory,
                    meshData->indices.data()));
            }

            // Copy data from staging buffers to device local buffers
            VkCommandBuffer copyCmd = m_context->vkDevice().createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            VkBufferCopy copyRegion = {};
            copyRegion.size = vertexBufferSize;
            vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, meshInstance->vertexBuffer->m_buffer, 1, &copyRegion);
            if (indexBufferSize > 0) {
                copyRegion.size = indexBufferSize;
                vkCmdCopyBuffer(copyCmd, indexStaging.buffer, meshInstance->indexBuffer->m_buffer, 1, &copyRegion);
            }
            m_context->vkDevice().flushCommandBuffer(copyCmd, m_context->vkDevice().m_TransferQueue, true);

            // Clean up staging buffers
            vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, vertexStaging.buffer, nullptr);
            vkFreeMemory(m_context->vkDevice().m_LogicalDevice, vertexStaging.memory, nullptr);
            if (indexBufferSize > 0) {
                vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, indexStaging.buffer, nullptr);
                vkFreeMemory(m_context->vkDevice().m_LogicalDevice, indexStaging.memory, nullptr);
            }
        }

        return meshInstance;
    }

    // The createMeshInstance method is as updated in section 1
}


/*
std::shared_ptr<MeshInstance> MeshResourceManager::createMeshInstance(const std::shared_ptr<MeshData> &meshData) {
    auto meshInstance = std::make_shared<MeshInstance>();
    meshInstance->topology = meshComponent.m_type == OBJ_FILE
                             ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                             : VK_PRIMITIVE_TOPOLOGY_POINT_LIST; // Set topology based on mesh data

    meshInstance->vertexCount = meshData->vertices.size();
    meshInstance->indexCount = meshData->indices.size();
    VkDeviceSize vertexBufferSize = meshData->vertices.size() * sizeof(Vertex);
    VkDeviceSize indexBufferSize = meshData->indices.size() * sizeof(uint32_t);
    meshInstance->m_type = meshComponent.m_type;
    if (meshComponent.m_type == CAMERA_GIZMO) {
        meshInstance->vertexCount = 21;
        return meshInstance;
    }

    if (!vertexBufferSize)
        return nullptr;

    struct StagingBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    } vertexStaging{}, indexStaging{};

    // Create staging buffers
    // Vertex m_DataPtr
    CHECK_RESULT(m_context->vkDevice().createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize,
            &vertexStaging.buffer,
            &vertexStaging.memory,
            meshData->vertices.data()))
    // Index m_DataPtr
    if (indexBufferSize > 0) {
        CHECK_RESULT(m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                meshData->indices.data()))
    }
    CHECK_RESULT(m_context->vkDevice().createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            meshInstance->vertexBuffer, vertexBufferSize, nullptr, "SceneRenderer:InitializeMesh:Vertex",
            m_context->getDebugUtilsObjectNameFunction()));
    // Index buffer
    if (indexBufferSize > 0) {
        CHECK_RESULT(m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                meshInstance->indexBuffer,
                indexBufferSize, nullptr, "SceneRenderer:InitializeMesh:Index",
                m_context->getDebugUtilsObjectNameFunction()));
    }
    // Copy from staging buffers
    VkCommandBuffer copyCmd = m_context->vkDevice().createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy copyRegion = {};
    copyRegion.size = vertexBufferSize;
    vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, meshInstance->vertexBuffer->m_buffer, 1, &copyRegion);
    if (indexBufferSize > 0) {
        copyRegion.size = indexBufferSize;
        vkCmdCopyBuffer(copyCmd, indexStaging.buffer, meshInstance->indexBuffer->m_buffer, 1, &copyRegion);
    }
    m_context->vkDevice().flushCommandBuffer(copyCmd, m_context->vkDevice().m_TransferQueue, true);
    vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, vertexStaging.buffer, nullptr);
    vkFreeMemory(m_context->vkDevice().m_LogicalDevice, vertexStaging.memory, nullptr);

    if (indexBufferSize > 0) {
        vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, indexStaging.buffer, nullptr);
        vkFreeMemory(m_context->vkDevice().m_LogicalDevice, indexStaging.memory, nullptr);
    }
    return meshInstance;
}
*/
