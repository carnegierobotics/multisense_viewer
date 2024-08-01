//
// Created by magnus on 7/29/24.
//


#include "Viewer/VkRender/Core/VulkanResourceManager.h"

namespace VkRender {
    VulkanResourceManager *VulkanResourceManager::instance = nullptr;

    void VulkanResourceManager::cleanup(bool onExit) {

        std::lock_guard<std::mutex> lock(resourceMutex);

        submitCommandBuffers();

        // Check if fences are signaled and execute corresponding cleanup functions
        for (auto it = m_deferredCleanupFunctions.begin(); it != m_deferredCleanupFunctions.end();) {
            VkResult result = vkGetFenceStatus(m_vulkanDevice->m_LogicalDevice, it->fence);
            if (result == VK_SUCCESS || onExit) {
                it->cleanupFunction();
                vkDestroyFence(m_vulkanDevice->m_LogicalDevice, it->fence, nullptr);
                it = m_deferredCleanupFunctions.erase(it);
            } else {
                ++it;
            }
        }

    }

    void VulkanResourceManager::submitCommandBuffers() {
        std::vector<VkSubmitInfo> submitInfos;
        // TODO Its possible to make batch submits, which could speed up a little bit
        // TODO Is m_TransferQueue the correct queue to use? Maybe use graphics queue as we are dealing with render resources
        for (auto &deferred: m_deferredCleanupFunctions) {
            VkCommandBuffer commandBuffer = m_vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            submitInfos.push_back(submitInfo);
            m_vulkanDevice->flushCommandBuffer(commandBuffer, m_vulkanDevice->m_TransferQueue, m_vulkanDevice->m_CommandPool, true,
                                               deferred.fence);
        }


    }

    std::once_flag VulkanResourceManager::initInstanceFlag;
    std::once_flag VulkanResourceManager::destroyInstanceFlag;

}
