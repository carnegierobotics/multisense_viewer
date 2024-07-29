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
#include <utility>
#include "CommandBuffer.h"
#include "VulkanDevice.h"

namespace VkRender {

    class VulkanResourceManager {
    public:
        static VulkanResourceManager& getInstance(VulkanDevice* device = VK_NULL_HANDLE, VmaAllocator allocator = VK_NULL_HANDLE) {
            std::call_once(initInstanceFlag, &VulkanResourceManager::initSingleton, device, allocator);
            return *instance;
        }
        static void destroyInstance() {
            std::call_once(destroyInstanceFlag, &VulkanResourceManager::cleanupSingleton);
        }

        ~VulkanResourceManager() {
            cleanup(true);
        }

        using CleanupFunction = std::function<void()>;

        void deferDeletion(CleanupFunction cleanupFunction, VkFence& fence) {
            std::lock_guard<std::mutex> lock(resourceMutex);
            m_deferredCleanupFunctions.push_back({std::move(cleanupFunction), fence});
        }

        void cleanup(bool onExit = false) {
            std::lock_guard<std::mutex> lock(resourceMutex);

            submitCommandBuffers();

            // Check if fences are signaled and execute corresponding cleanup functions
            for (auto it = m_deferredCleanupFunctions.begin(); it != m_deferredCleanupFunctions.end();) {
                VkResult result = vkGetFenceStatus(m_device->m_LogicalDevice, it->fence);
                if (result == VK_SUCCESS || onExit) {
                    it->cleanupFunction();
                    vkDestroyFence(m_device->m_LogicalDevice, it->fence, nullptr);
                    it = m_deferredCleanupFunctions.erase(it);
                } else {
                    ++it;
                }
            }
        }

    private:
        VulkanResourceManager(VulkanDevice* device, VmaAllocator allocator)
                : m_device(device), m_allocator(allocator) {}

        static void initSingleton(VulkanDevice* device, VmaAllocator allocator) {
            instance = new VulkanResourceManager(device, allocator);
        }

        static void cleanupSingleton() {
            delete instance;
            instance = nullptr;
        }

        void submitCommandBuffers() {
            std::vector<VkSubmitInfo> submitInfos;
            // TODO Its possible to make batch submits, which could speed up a little bit
            // TODO Is m_TransferQueue the correct queue to use? Maybe use graphics queue as we are dealing with render resources
            for (auto& deferred : m_deferredCleanupFunctions) {
                VkCommandBuffer commandBuffer = m_device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

                VkSubmitInfo submitInfo = {};
                submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &commandBuffer;
                submitInfos.push_back(submitInfo);
                m_device->flushCommandBuffer(commandBuffer, m_device->m_TransferQueue, m_device->m_CommandPool, true, deferred.fence);
            }

        }

        static VulkanResourceManager* instance;
        static std::once_flag initInstanceFlag;
        static std::once_flag destroyInstanceFlag;
        std::mutex resourceMutex;

        VulkanDevice* m_device;
        VmaAllocator m_allocator = VK_NULL_HANDLE;

        struct DeferredCleanup {
            CleanupFunction cleanupFunction;
            VkFence fence;
        };

        std::deque<DeferredCleanup> m_deferredCleanupFunctions;
    };

}

#endif //MULTISENSE_VIEWER_VULKANRESOURCEMANAGER_H
