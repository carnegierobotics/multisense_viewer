//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_OBJMODELCOMPONENT
#define MULTISENSE_VIEWER_OBJMODELCOMPONENT

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/detail/type_quat.hpp>
#include <glm/fwd.hpp>
#include <vulkan/vulkan_core.h>
#include <cfloat>

#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Texture.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/CommandBuffer.h"

namespace VkRender {
    struct OBJModelComponent {
    public:
        OBJModelComponent() = default;

        OBJModelComponent(const OBJModelComponent &) = default;

        OBJModelComponent &operator=(const OBJModelComponent &other) {
            return *this;
        }


        OBJModelComponent(const std::filesystem::path &modelPath, VulkanDevice *dev) {
            device = dev;
            loadModel(modelPath);
            loadTexture(modelPath);
        }

        ~OBJModelComponent() {
            if (vertices.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device->m_LogicalDevice, vertices.buffer, nullptr);
            }
            if (vertices.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device->m_LogicalDevice, vertices.memory, nullptr);
            }
            if (indices.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device->m_LogicalDevice, indices.buffer, nullptr);
            }
            if (indices.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device->m_LogicalDevice, indices.memory, nullptr);
            }
        }

        Texture2D objTexture; // TODO Possibly make more empty textures to match our triple buffering?
        void draw(CommandBuffer *commandBuffer, uint32_t cbIndex);

    private:
        void loadModel(std::filesystem::path modelPath);

        void loadTexture(std::filesystem::path texturePath);

        struct Vertices {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            uint32_t vertexCount = 0;
        };
        struct Indices {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            uint32_t indexCount = 0;
        };

        struct LoaderInfo {
            uint32_t *indexBuffer;
            VkRender::Vertex *vertexBuffer;
        };

    private:
        VulkanDevice *device;
        Indices indices{};
        Vertices vertices{};

    };
};


#endif //MULTISENSE_VIEWER_OBJMODELCOMPONENT
