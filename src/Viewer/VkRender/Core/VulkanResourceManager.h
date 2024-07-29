//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_VULKANRESOURCEMANAGER_H
#define MULTISENSE_VIEWER_VULKANRESOURCEMANAGER_H



#include <mutex>
#include <deque>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <functional>

namespace VkRender {

    class VulkanResourceManager {
    public:
        static VulkanResourceManager& getInstance(VkDevice logicalDevice = VK_NULL_HANDLE, VmaAllocator allocator = VK_NULL_HANDLE) {
            std::call_once(initInstanceFlag, &VulkanResourceManager::initSingleton, logicalDevice, allocator);
            return *instance;
        }

        ~VulkanResourceManager() {
            cleanup();
        }

        using CleanupFunction = std::function<void()>;

        void deferDeletion(CleanupFunction cleanupFunction, VkFence fence) {
            std::lock_guard<std::mutex> lock(resourceMutex);
            m_deferredCleanupFunctions.push_back({cleanupFunction, fence});
        }

        void cleanup() {
            std::lock_guard<std::mutex> lock(resourceMutex);

            // Check if fences are signaled and execute corresponding cleanup functions
            for (auto it = m_deferredCleanupFunctions.begin(); it != m_deferredCleanupFunctions.end();) {
                VkResult result = vkGetFenceStatus(m_logicalDevice, it->fence);
                if (result == VK_SUCCESS) {
                    it->cleanupFunction();
                    vkDestroyFence(m_logicalDevice, it->fence, nullptr);
                    it = m_deferredCleanupFunctions.erase(it);
                } else {
                    ++it;
                }
            }
        }

    private:
        VulkanResourceManager(VkDevice logicalDevice, VmaAllocator allocator)
                : m_logicalDevice(logicalDevice), m_allocator(allocator) {}

        static void initSingleton(VkDevice logicalDevice, VmaAllocator allocator) {
            instance = new VulkanResourceManager(logicalDevice, allocator);
        }

        static VulkanResourceManager* instance;
        static std::once_flag initInstanceFlag;
        std::mutex resourceMutex;

        VkDevice m_logicalDevice;
        VmaAllocator m_allocator;

        struct DeferredCleanup {
            CleanupFunction cleanupFunction;
            VkFence fence;
        };

        std::deque<DeferredCleanup> m_deferredCleanupFunctions;
    };

}

#endif //MULTISENSE_VIEWER_VULKANRESOURCEMANAGER_H
