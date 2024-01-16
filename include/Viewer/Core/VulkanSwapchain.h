/**
 * @file: MultiSense-Viewer/include/Viewer/Core/VulkanSwapchain.h
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
 *   2021-09-4, mgjerde@carnegierobotics.com, Created file.
 **/
#ifndef MULTISENSE_VULKANSWAPCHAIN_H
#define MULTISENSE_VULKANSWAPCHAIN_H


#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#ifdef WIN32
    #ifdef APIENTRY
        #undef APIENTRY
    #endif
#endif
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <vector>
#include <cassert>

#include "Viewer/Core/RenderDefinitions.h"

typedef struct SwapChainBuffers {
    VkImage image;
    VkImageView view;
} SwapChainBuffer;

class VulkanSwapchain {
private:
    VkInstance instance{};
    VkDevice device{};
    VkPhysicalDevice physicalDevice{};
public:
    VkSurfaceKHR surface{};
    VkFormat colorFormat{};
    VkColorSpaceKHR colorSpace{};
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    uint32_t imageCount=0;
    uint32_t swapChainImagesUsed = 0;
    std::vector<VkImage> images{};
    std::vector<SwapChainBuffer> buffers{};
    uint32_t queueNodeIndex = UINT32_MAX;

    void create(uint32_t* width, uint32_t* height, bool vsync = false);
    VkResult acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t* imageIndex);
    VkResult queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore = VK_NULL_HANDLE);
    void cleanup();

    VulkanSwapchain(VkRender::SwapChainCreateInfo info, uint32_t *width, uint32_t *height);
};


#endif //MULTISENSE_VULKANSWAPCHAIN_H
