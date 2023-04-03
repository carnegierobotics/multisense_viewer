/**
 * @file: MultiSense-Viewer/src/Core/VulkanRenderer.cpp
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
 *   2021-8-28, mgjerde@carnegierobotics.com, Created file.
 **/

#include <stb_image.h>

#include "Viewer/Core/VulkanRenderer.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Core/Validation.h"
#include "Viewer/Tools/Populate.h"

#ifdef WIN32
#include <strsafe.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#endif
namespace VkRender {
    VulkanRenderer::VulkanRenderer(const std::string &title, bool enableValidation) {
        settings.validation = enableValidation;
        // Create window instance
        // boilerplate stuff (ie. basic window setup, initialize OpenGL) occurs in abstract class
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        m_Width = 1280;
        m_Height = 720;
        window = glfwCreateWindow(m_Width, m_Height, title.c_str(), nullptr, nullptr);
        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, this);
        glfwSetKeyCallback(window, VulkanRenderer::keyCallback);
        glfwSetWindowSizeCallback(window, VulkanRenderer::resizeCallback);
        glfwSetMouseButtonCallback(window, VulkanRenderer::mouseButtonCallback);
        glfwSetCursorPosCallback(window, VulkanRenderer::cursorPositionCallback);
        glfwSetScrollCallback(window, VulkanRenderer::mouseScrollCallback);
        glfwSetCharCallback(window, VulkanRenderer::charCallback);

        GLFWimage images[1];
        std::string fileName = Utils::getAssetsPath().append("Textures/CRL96x96.png").string();
        images[0].pixels = stbi_load(fileName.c_str(), &images[0].width, &images[0].height, nullptr, 4); //rgba channels
        if (!images[0].pixels) {
            throw std::runtime_error("Failed to load window icon: " + fileName);
        }
        glfwSetWindowIcon(window, 1, images);
        stbi_image_free(images[0].pixels);

    }

    VkResult VulkanRenderer::createInstance(bool enableValidation) {
        settings.validation = enableValidation;
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = m_Name.c_str();
        appInfo.pEngineName = m_Name.c_str();
        appInfo.apiVersion = apiVersion;
        pLogger->info("Setting up vulkan with API Version: {}.{}.{} Minimum recommended version to use is 1.2.0",
                      VK_API_VERSION_MAJOR(apiVersion), VK_API_VERSION_MINOR(apiVersion),
                      VK_API_VERSION_PATCH(apiVersion));
        // Get extensions supported by the instance
        enabledInstanceExtensions = Validation::getRequiredExtensions(settings.validation);
        // Check if extensions are supported
        if (!Validation::checkInstanceExtensionSupport(enabledInstanceExtensions))
            throw std::runtime_error("Instance Extensions not supported");
        VkInstanceCreateInfo instanceCreateInfo = {};
        instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.pNext = NULL;
        instanceCreateInfo.pApplicationInfo = &appInfo;
        if (!enabledInstanceExtensions.empty()) {
            if (settings.validation) {
                enabledInstanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }
            instanceCreateInfo.enabledExtensionCount = (uint32_t) enabledInstanceExtensions.size();
            instanceCreateInfo.ppEnabledExtensionNames = enabledInstanceExtensions.data();
        }
        const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
        // The VK_LAYER_KHRONOS_validation contains all current validation functionality.
        if (settings.validation) {
            // Check if this layer is available at instance level
            if (Validation::checkValidationLayerSupport(validationLayers)) {
                instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
                instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                pLogger->info("Enabling Validation Layers");
            } else {
                std::cerr << "Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled\n";
                pLogger->info("Disabled Validation Layers since it was not found");

            }
        }
        return vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
    }

    bool VulkanRenderer::initVulkan() {
        // Create Instance
        VkResult err = createInstance(settings.validation);
        if (err) {
            throw std::runtime_error("Could not create Vulkan instance");
        }
        pLogger->info("Vulkan Instance successfully created");
        // If requested, we flashing the default validation layers for debugging
// If requested, we flashing the default validation layers for debugging
        if (settings.validation) {
            // The report flags determine what type of messages for the layers will be displayed
            // For validating (debugging) an application the error and warning bits should suffice
            // Additional flags include performance info, loader and layer debug messages, etc.
            VkDebugUtilsMessengerCreateInfoEXT createInfo;
            Validation::populateDebugMessengerCreateInfo(createInfo);
            if (Validation::CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugUtilsMessenger) !=
                VK_SUCCESS) {
                throw std::runtime_error("failed to set up debug messenger!");
            }
        }
        // Get list of devices and capabilities of each m_Device
        uint32_t gpuCount = 0;
        err = vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
        if (err != VK_SUCCESS or gpuCount == 0) {
            throw std::runtime_error("No m_Device with vulkan support found");
        }
        // Enumerate devices
        std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
        err = vkEnumeratePhysicalDevices(instance, &gpuCount, physicalDevices.data());
        if (err != VK_SUCCESS) {
            throw std::runtime_error("Could not enumerate physical devices");
        }
        // Select physical m_Device to be used for the Vulkan example
        // Defaults to the first m_Device unless anything else specified
        physicalDevice = pickPhysicalDevice(physicalDevices);
        // If pyshyical m_Device supports vulkan version > apiVersion then create new instance with this version.
        // Store m_Properties (including limits), m_Features and memory m_Properties of the physical m_Device (so that examples can check against them)
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
        vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

        VkPhysicalDeviceSamplerYcbcrConversionFeatures features;
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
        features.pNext = nullptr;
        // Derived examples can override this to set actual m_Features (based on above readings) to flashing for logical m_Device creation
        addDeviceFeatures();
        VkPhysicalDeviceFeatures2 features2;
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features;
        features2.features = deviceFeatures;

        vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

        // If available then: Add KHR_SAMPLER_YCBCR For Color camera data m_Format.
        if (features.samplerYcbcrConversion) {
            enabledDeviceExtensions.push_back(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
            pLogger->info("Enabling YCBCR Sampler Extension");
        } else {
            pLogger->error("YCBCR Sampler support not found!");
        }

        // Vulkan m_Device creation
        // This is firstUpdate by a separate class that gets a logical m_Device representation
        // and encapsulates functions related to a m_Device
        vulkanDevice = std::make_unique<VulkanDevice>(physicalDevice);
        err = vulkanDevice->createLogicalDevice(enabledFeatures, enabledDeviceExtensions, &features);
        if (err != VK_SUCCESS)
            throw std::runtime_error("Failed to create logical device");

        device = vulkanDevice->m_LogicalDevice;
        // Get a graphics queue from the m_Device
        vkGetDeviceQueue(device, vulkanDevice->m_QueueFamilyIndices.graphics, 0, &queue);
        // Find a suitable depth m_Format
        depthFormat = Utils::findDepthFormat(physicalDevice);
        // Create synchronization Objects
        VkSemaphoreCreateInfo semaphoreCreateInfo = Populate::semaphoreCreateInfo();
        semaphoreCreateInfo.flags =
                // Create a semaphore used to synchronize m_Image presentation
                // Ensures that the m_Image is displayed before we start submitting new commands to the queue
        err = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores.presentComplete);
        if (err)
            throw std::runtime_error("Failed to create semaphore");
        // Create a semaphore used to synchronize command submission
        // Ensures that the m_Image is not presented until all commands have been submitted and executed
        err = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores.renderComplete);
        if (err)
            throw std::runtime_error("Failed to create semaphore");
        // Set up submit info structure
        // Semaphores will stay the same during application lifetime
        // Command buffer submission info is set by each example
        submitInfo = Populate::submitInfo();
        submitInfo.pWaitDstStageMask = &submitPipelineStages;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &semaphores.presentComplete;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &semaphores.renderComplete;
        return true;
    }

    VulkanRenderer::~VulkanRenderer() {
        // CleanUP all vulkan resources
        swapchain->cleanup();
        // Object picking resources
        vkDestroyRenderPass(device, selection.renderPass, nullptr);
        vkDestroyFramebuffer(device, selection.frameBuffer, nullptr);
        vkDestroyImage(device, selection.colorImage, nullptr);
        vkDestroyImage(device, selection.depthImage, nullptr);
        vkDestroyImageView(device, selection.colorView, nullptr);
        vkDestroyImageView(device, selection.depthView, nullptr);
        vkFreeMemory(device, selection.colorMem, nullptr);
        vkFreeMemory(device, selection.depthMem, nullptr);
        // VulkanRenderer resources
        vkDestroyImage(device, depthStencil.image, nullptr);
        vkDestroyImageView(device, depthStencil.view, nullptr);
        vkFreeMemory(device, depthStencil.mem, nullptr);
        vkDestroyCommandPool(device, cmdPool, nullptr);
        for (auto &fence: waitFences) {
            vkDestroyFence(device, fence, nullptr);
        }
        vkDestroyRenderPass(device, renderPass, nullptr);
        for (auto &fb: frameBuffers) {
            vkDestroyFramebuffer(device, fb, nullptr);
        }
        vkDestroyPipelineCache(device, pipelineCache, nullptr);
        vkDestroySemaphore(device, semaphores.presentComplete, nullptr);
        vkDestroySemaphore(device, semaphores.renderComplete, nullptr);
        if (settings.validation)
            Validation::DestroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, nullptr);

        vulkanDevice.reset(); //Call to destructor for smart pointer destroy logical m_Device before instance
        vkDestroyInstance(instance, nullptr);
        // CleanUp GLFW window
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void VulkanRenderer::addDeviceFeatures() {
    }

    void VulkanRenderer::viewChanged() {

    }


    void VulkanRenderer::mouseMoved(float x, float y, bool &handled) {
        mousePos = glm::vec2(x, y);
        handled = true;
    }

    void VulkanRenderer::windowResized() {

    }

    void VulkanRenderer::buildCommandBuffers() {

    }

    void VulkanRenderer::setupDepthStencil() {
        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = depthFormat;
        imageCI.extent = {m_Width, m_Height, 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        VkResult result = vkCreateImage(device, &imageCI, nullptr, &depthStencil.image);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image");

        VkMemoryRequirements memReqs{};
        vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);

        VkMemoryAllocateInfo memAllloc = Populate::memoryAllocateInfo();
        memAllloc.allocationSize = memReqs.size;
        memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        result = vkAllocateMemory(device, &memAllloc, nullptr, &depthStencil.mem);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth m_Image memory");
        result = vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth m_Image memory");

        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.image = depthStencil.image;
        imageViewCI.format = depthFormat;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        result = vkCreateImageView(device, &imageViewCI, nullptr, &depthStencil.view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image m_View");
    }

    void VulkanRenderer::setupMainFramebuffer() {
        // Depth/Stencil attachment is the same for all frame buffers
        std::array<VkImageView, 2> attachments{};
        attachments[1] = depthStencil.view;
        VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(m_Width, m_Height,
                                                                                        attachments.data(),
                                                                                        attachments.size(),
                                                                                        renderPass);
        frameBuffers.resize(swapchain->imageCount);
        for (uint32_t i = 0; i < frameBuffers.size(); i++) {
            attachments[0] = swapchain->buffers[i].view;
            VkResult result = vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &frameBuffers[i]);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");
        }
    }

    void VulkanRenderer::setupRenderPass() {
        {
            std::array<VkAttachmentDescription, 2> attachments{};
            // Color attachment
            attachments[0].format = swapchain->colorFormat;
            attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
            attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            // Depth attachment
            attachments[1].format = depthFormat;
            attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
            attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentReference colorReference{};
            colorReference.attachment = 0;
            colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentReference depthReference{};
            depthReference.attachment = 1;
            depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpassDescription{};
            subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpassDescription.colorAttachmentCount = 1;
            subpassDescription.pColorAttachments = &colorReference;
            subpassDescription.pDepthStencilAttachment = &depthReference;
            subpassDescription.inputAttachmentCount = 0;
            subpassDescription.pInputAttachments = nullptr;
            subpassDescription.preserveAttachmentCount = 0;
            subpassDescription.pPreserveAttachments = nullptr;
            subpassDescription.pResolveAttachments = nullptr;

            // Subpass dependencies for layout transitions
            std::array<VkSubpassDependency, 2> dependencies{};

            dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[0].dstSubpass = 0;
            dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            dependencies[1].srcSubpass = 0;
            dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            VkRenderPassCreateInfo renderPassInfo = Populate::renderPassCreateInfo();
            renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            renderPassInfo.pAttachments = attachments.data();
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpassDescription;
            renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
            renderPassInfo.pDependencies = dependencies.data();

            VkResult result = (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to create render pass");
        }

        /** CREATE SECONDARY RENDER PASS */
        {
            // Setup picking render pass
            std::array<VkAttachmentDescription, 2> attachments = {};
            // Color attachment
            attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
            attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
            attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            // Depth attachment
            attachments[1].format = depthFormat;
            attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
            attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[1].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

            VkAttachmentReference colorReference = {};
            colorReference.attachment = 0;
            colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentReference depthReference = {};
            depthReference.attachment = 1;
            depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpassDescription = {};
            subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpassDescription.colorAttachmentCount = 1;
            subpassDescription.pColorAttachments = &colorReference;
            subpassDescription.pDepthStencilAttachment = &depthReference;
            subpassDescription.inputAttachmentCount = 0;
            subpassDescription.pInputAttachments = nullptr;
            subpassDescription.preserveAttachmentCount = 0;
            subpassDescription.pPreserveAttachments = nullptr;
            subpassDescription.pResolveAttachments = nullptr;

            // Subpass dependencies for layout transitions
            std::array<VkSubpassDependency, 2> dependencies{};

            dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[0].dstSubpass = 0;
            dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            dependencies[1].srcSubpass = 0;
            dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            VkRenderPassCreateInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            renderPassInfo.pAttachments = attachments.data();
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpassDescription;
            renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
            renderPassInfo.pDependencies = dependencies.data();

            VkResult result = (vkCreateRenderPass(device, &renderPassInfo, nullptr, &selection.renderPass));
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to create render pass");
        }
    }

    void VulkanRenderer::createCommandPool() {
        VkCommandPoolCreateInfo cmdPoolInfo = Populate::commandPoolCreateInfo();
        cmdPoolInfo.queueFamilyIndex = swapchain->queueNodeIndex;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkResult result = vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create command pool");
    }

    void VulkanRenderer::createCommandBuffers() {
        // Create one command buffer for each swap chain m_Image and reuse for rendering
        drawCmdBuffers.resize(swapchain->imageCount);

        VkCommandBufferAllocateInfo cmdBufAllocateInfo =
                Populate::commandBufferAllocateInfo(
                        cmdPool,
                        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                        static_cast<uint32_t>(drawCmdBuffers.size()));

        VkResult result = vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, drawCmdBuffers.data());
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create command pool");
    }

    void VulkanRenderer::createSynchronizationPrimitives() {
        // Wait fences to sync command buffer access
        VkFenceCreateInfo fenceCreateInfo = Populate::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
        waitFences.resize(drawCmdBuffers.size());
        for (auto &fence: waitFences) {
            VkResult result = vkCreateFence(device, &fenceCreateInfo, nullptr, &fence);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to create command pool");
        }
    }


    void VulkanRenderer::createPipelineCache() {
        VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
        pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        VkResult result = vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create command pool");
    }


    void VulkanRenderer::prepare() {
        VkRender::SwapChainCreateInfo info{};
        info.instance = instance;
        info.pWindow = window;
        info.physicalDevice = physicalDevice;
        info.device = device;
        swapchain = std::make_unique<VulkanSwapchain>(info, &m_Width, &m_Height);

        createCommandPool();
        createCommandBuffers();
        createSynchronizationPrimitives();
        setupDepthStencil();
        setupRenderPass();
        createPipelineCache();
        setupMainFramebuffer();

        pLogger->info("Initialized Renderer backend");

        startTime = std::chrono::system_clock::now();


    }


    void VulkanRenderer::destroyCommandBuffers() {
        vkFreeCommandBuffers(device, cmdPool, static_cast<uint32_t>(drawCmdBuffers.size()), drawCmdBuffers.data());
    }

    void VulkanRenderer::windowResize() {
        if (!backendInitialized) {
            return;
        }
        backendInitialized = false;

        // Ensure all operations on the m_Device have been finished before destroying resources
        vkQueueWaitIdle(queue);
        vkDeviceWaitIdle(device);

        glfwGetFramebufferSize(window, reinterpret_cast<int *>(&m_Width), reinterpret_cast<int *>(&m_Height));
        while (m_Width == 0 || m_Height == 0) {
            glfwGetFramebufferSize(window, reinterpret_cast<int *>(&m_Width), reinterpret_cast<int *>(&m_Height));
            glfwWaitEvents();
        }
        // Recreate swap chain
        swapchain->create(&m_Width, &m_Height, settings.vsync);

        // Recreate the frame buffers
        vkDestroyImageView(device, depthStencil.view, nullptr);
        vkDestroyImage(device, depthStencil.image, nullptr);
        vkFreeMemory(device, depthStencil.mem, nullptr);
        setupDepthStencil();
        for (uint32_t i = 0; i < frameBuffers.size(); i++) {
            vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
        }

        for (const auto& fence : waitFences) {
            vkDestroyFence(device, fence, nullptr);
        }
        setupMainFramebuffer();

        // Maybe resize overlay too
        // --- Missing code snippet ---

        // Command buffers need to be recreated as they may store
        // references to the recreated frame buffer
        destroyCommandBuffers();
        createCommandBuffers();
        createSynchronizationPrimitives();
        buildCommandBuffers();
        vkDeviceWaitIdle(device);

        if ((m_Width > 0.0f) && (m_Height > 0.0f)) {
            camera.updateAspectRatio((float) m_Width / (float) m_Height);
        }
        pLogger->info("Window Resized. New size is: {} x {}", m_Width, m_Height);

        // Notify derived class
        windowResized();
        viewChanged();

        backendInitialized = true;
    }


    void VulkanRenderer::renderLoop() {
        destWidth = m_Width;
        destHeight = m_Height;
        auto graphLastTimestamp = std::chrono::high_resolution_clock::now();

        while (!glfwWindowShouldClose(window)) {
            auto tStart = std::chrono::high_resolution_clock::now();
            frameID++; // First frame will have id 1.
            glfwPollEvents();
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - startTime;
            runTime = elapsed_seconds.count();
            /** Give ImGui Reference to this frame's input events **/
            ImGuiIO &io = ImGui::GetIO();
            io.DisplaySize = ImVec2((float) m_Width, (float) m_Height);
            io.DeltaTime = frameTimer;
            io.WantCaptureMouse = true;
            io.MousePos = ImVec2(mousePos.x, mousePos.y);
            io.MouseDown[0] = mouseButtons.left;
            io.MouseDown[1] = mouseButtons.right;
            input.lastKeyPress = keyPress;
            input.action = keyAction;
            prepareFrame();
            /** Call Renderer's render function **/
            render();
            /** Reset some variables for next frame **/
            submitFrame();
            keyPress = -1;
            keyAction = -1;
            io.MouseWheel = 0;
            mouseButtons.dx = 0;
            mouseButtons.dy = 0;
            mouseButtons.action = -1;
            /** FrameTiming **/
            auto tEnd = std::chrono::high_resolution_clock::now();
            frameCounter++;
            float fpsTimer = std::chrono::duration<float, std::milli>(tEnd - graphLastTimestamp).count();
            if (fpsTimer > 1000.0f) {
                lastFPS = (float) frameCounter * (1000.0f / fpsTimer);
                frameCounter = 0;
                graphLastTimestamp = tEnd;
            }
            auto tDiff = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - tStart).count();
            frameTimer = (float) tDiff / 1000.0f;
            camera.update(frameTimer);
        }
        // Flush m_Device to make sure all resources can be freed before we start cleanup
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
    }

    void VulkanRenderer::prepareFrame() {
        // Acquire the next m_Image from the swap chain
        VkResult result = swapchain->acquireNextImage(semaphores.presentComplete, &currentBuffer);
        // Recreate the swapchain if it's no longer compatible with the surface (OUT_OF_DATE) or no longer optimal for presentation (SUBOPTIMAL)
        if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
            Log::Logger::getInstance()->info("SwapChain no longer compatible on acquire next image. Recreating..");
            windowResize();
            VkResult res = swapchain->acquireNextImage(semaphores.presentComplete, &currentBuffer);
            if (res != VK_SUCCESS)
                Log::Logger::getInstance()->error("Suboptimal Surface: Failed to acquire next m_Image after windoResize. VkResult: {}", std::to_string(result));

        } else if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to acquire next m_Image in prepareFrame. VkResult: " + std::to_string(result));

        // Use a fence to wait until the command buffer has finished execution before using it again
        result = vkWaitForFences(device, 1, &waitFences[currentBuffer], VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to wait for fence");
        result = vkResetFences(device, 1, &waitFences[currentBuffer]);
        if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to reset fence");
    }

    void VulkanRenderer::submitFrame() {
        vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentBuffer]);

        VkResult result = swapchain->queuePresent(queue, currentBuffer, semaphores.renderComplete);
        if (!((result == VK_SUCCESS) || (result == VK_SUBOPTIMAL_KHR))) {
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                // Swap chain is no longer compatible with the surface and needs to be recreated
                Log::Logger::getInstance()->info("SwapChain no longer compatible on queue present. Recreating..");
                windowResize();
                return;
            }
        } else if (result != VK_SUCCESS) {
            Log::Logger::getInstance()->error("Suboptimal Surface: Failed to acquire next m_Image. VkResult: {}", std::to_string(result));
        }
        if (vkQueueWaitIdle(queue) != VK_SUCCESS)
        throw std::runtime_error("Failed to wait for Queue Idle. This should not happen and may indicate lost GPU instance. Shutting down . VkResult: " + std::to_string(result));
    }

/** CALLBACKS **/
    void VulkanRenderer::setWindowSize(uint32_t _width, uint32_t _height) {
        if (frameID > 1) {
            destWidth = _width;
            destHeight = _height;
            //Log::Logger::getInstance()->info("New window size was set. Recreating..");
            //windowResize();
        }

    }

    void VulkanRenderer::resizeCallback(GLFWwindow *window, int width, int height) {
        auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
        if (width > 0 || height > 0) {
            if (myApp->destWidth != width && myApp->destHeight != height)
                myApp->setWindowSize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
        }
    }

    void VulkanRenderer::charCallback(GLFWwindow *window, unsigned int codepoint) {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
        ImGuiIO &io = ImGui::GetIO();
        io.AddInputCharacter((unsigned short) codepoint);
        DISABLE_WARNING_POP

    }


    void VulkanRenderer::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
        auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
        if ((key == GLFW_KEY_ESCAPE) && action == GLFW_PRESS) {
            myApp->pLogger->info("Escape key registered. Closing program..");
            glfwSetWindowShouldClose(window, true);
        }
        ImGuiIO &io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiKey_ModShift, (mods & GLFW_MOD_SHIFT) != 0);
        io.AddKeyEvent(ImGuiKey_ModAlt, (mods & GLFW_MOD_ALT) != 0);
        io.AddKeyEvent(ImGuiKey_ModSuper, (mods & GLFW_MOD_SUPER) != 0);
        io.AddKeyEvent(ImGuiKey_LeftCtrl, (mods & GLFW_MOD_CONTROL) != 0);

        key = ImGui_ImplGlfw_TranslateUntranslatedKey(key, scancode);
        ImGuiKey imgui_key = ImGui_ImplGlfw_KeyToImGuiKey(key);
        io.AddKeyEvent(imgui_key, (action == GLFW_PRESS) || (action == GLFW_REPEAT));
        myApp->keyPress = key;
        myApp->keyAction = action;

#ifdef WIN32
        if ((mods & GLFW_MOD_CONTROL) != 0 && key == GLFW_KEY_V){
            myApp->clipboard();
        }
#endif

        if (action == GLFW_PRESS) {
            switch (key) {
                case GLFW_KEY_W:
                case GLFW_KEY_UP:
                    myApp->camera.keys.up = true;
                    break;
                case GLFW_KEY_S:
                case GLFW_KEY_DOWN:
                    myApp->camera.keys.down = true;
                    break;
                case GLFW_KEY_A:
                case GLFW_KEY_LEFT:
                    myApp->camera.keys.left = true;
                    break;
                case GLFW_KEY_D:
                case GLFW_KEY_RIGHT:
                    myApp->camera.keys.right = true;
                default:
                    break;
            }
        }
        if (action == GLFW_RELEASE) {
            switch (key) {
                case GLFW_KEY_W:
                case GLFW_KEY_UP:
                    myApp->camera.keys.up = false;
                    break;
                case GLFW_KEY_S:
                case GLFW_KEY_DOWN:
                    myApp->camera.keys.down = false;
                    break;
                case GLFW_KEY_A:
                case GLFW_KEY_LEFT:
                    myApp->camera.keys.left = false;
                    break;
                case GLFW_KEY_D:
                case GLFW_KEY_RIGHT:
                    myApp->camera.keys.right = false;
                default:
                    break;
            }
        }
    }

    void VulkanRenderer::handleMouseMove(float x, float y) {
        bool handled = false;
        if (settings.overlay) {
            ImGuiIO &io = ImGui::GetIO();
            io.WantCaptureMouse = true;
        }

        mouseMoved(x, y, handled);
        viewChanged();
    }

    void VulkanRenderer::cursorPositionCallback(GLFWwindow *window, double xPos, double yPos) {
        auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
        myApp->handleMouseMove((float) xPos, (float) yPos);

    }

    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

    void VulkanRenderer::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
        auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));

        if (action == GLFW_PRESS) {
            switch (button) {
                case GLFW_MOUSE_BUTTON_RIGHT:
                    myApp->mouseButtons.right = true;
                    break;
                case GLFW_MOUSE_BUTTON_MIDDLE:
                    myApp->mouseButtons.middle = true;
                    break;
                case GLFW_MOUSE_BUTTON_LEFT:
                    myApp->mouseButtons.left = true;
                    break;
            }
            myApp->mouseButtons.action = GLFW_PRESS;
        }
        if (action == GLFW_RELEASE) {
            switch (button) {
                case GLFW_MOUSE_BUTTON_RIGHT:
                    myApp->mouseButtons.right = false;
                    break;
                case GLFW_MOUSE_BUTTON_MIDDLE:
                    myApp->mouseButtons.middle = false;
                    break;
                case GLFW_MOUSE_BUTTON_LEFT:
                    myApp->mouseButtons.left = false;
                    break;
            }
            myApp->mouseButtons.action = GLFW_RELEASE;
        }
    }

    void VulkanRenderer::mouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
        auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
        ImGuiIO &io = ImGui::GetIO();
        myApp->mouseScroll((float) yoffset);
        io.MouseWheel += 0.5f * (float) yoffset;
    }

    DISABLE_WARNING_POP

    VkPhysicalDevice VulkanRenderer::pickPhysicalDevice(std::vector<VkPhysicalDevice> devices) const {
        if (devices.empty())
            throw std::runtime_error("No physical devices available");
        for (auto &d: devices) {
            VkPhysicalDeviceProperties properties{};
            VkPhysicalDeviceFeatures features{};
            VkPhysicalDeviceMemoryProperties memoryProperties{};
            vkGetPhysicalDeviceProperties(d, &properties);
            vkGetPhysicalDeviceFeatures(d, &features);
            vkGetPhysicalDeviceMemoryProperties(d, &memoryProperties);
            pLogger->info("Found physical d: {}, ", properties.deviceName);

            // Search for a discrete GPU and prefer this one
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                pLogger->info("Picked Discrete GPU. Name: {}, ", properties.deviceName);
                return d;
            }
        }
        // If no discrete GPU were found just return the first m_Device found
        return devices[0];
    }

    int VulkanRenderer::ImGui_ImplGlfw_TranslateUntranslatedKey(int key, int scancode) {
#if GLFW_HAS_GET_KEY_NAME && !defined(__EMSCRIPTEN__)
        // GLFW 3.1+ attempts to "untranslate" keys, which goes the opposite of what every other framework does, making using lettered shortcuts difficult.
        // (It had reasons to do so: namely GLFW is/was more likely to be used for WASD-type game controls rather than lettered shortcuts, but IHMO the 3.1 change could have been done differently)
        // See https://github.com/glfw/glfw/issues/1502 for details.
        // Adding a workaround to undo this (so our keys are translated->untranslated->translated, likely a lossy process).
        // This won't cover edge cases but this is at least going to cover common cases.
        if (key >= GLFW_KEY_KP_0 && key <= GLFW_KEY_KP_EQUAL)
            return key;
        const char* key_name = glfwGetKeyName(key, scancode);
        if (key_name && key_name[0] != 0 && key_name[1] == 0)
        {
            const char char_names[] = "`-=[]\\,;\'./";
            const int char_keys[] = { GLFW_KEY_GRAVE_ACCENT, GLFW_KEY_MINUS, GLFW_KEY_EQUAL, GLFW_KEY_LEFT_BRACKET, GLFW_KEY_RIGHT_BRACKET, GLFW_KEY_BACKSLASH, GLFW_KEY_COMMA, GLFW_KEY_SEMICOLON, GLFW_KEY_APOSTROPHE, GLFW_KEY_PERIOD, GLFW_KEY_SLASH, 0 };
            IM_ASSERT(IM_ARRAYSIZE(char_names) == IM_ARRAYSIZE(char_keys));
            if (key_name[0] >= '0' && key_name[0] <= '9')               { key = GLFW_KEY_0 + (key_name[0] - '0'); }
            else if (key_name[0] >= 'A' && key_name[0] <= 'Z')          { key = GLFW_KEY_A + (key_name[0] - 'A'); }
            else if (const char* p = strchr(char_names, key_name[0]))   { key = char_keys[p - char_names]; }
        }
        // if (action == GLFW_PRESS) printf("key %d scancode %d m_Name '%s'\n", key, scancode, key_name);
#else
        IM_UNUSED(scancode);
#endif
        return key;
    }

    ImGuiKey VulkanRenderer::ImGui_ImplGlfw_KeyToImGuiKey(int key) {
        switch (key) {
            case GLFW_KEY_TAB:
                return ImGuiKey_Tab;
            case GLFW_KEY_LEFT:
                return ImGuiKey_LeftArrow;
            case GLFW_KEY_RIGHT:
                return ImGuiKey_RightArrow;
            case GLFW_KEY_UP:
                return ImGuiKey_UpArrow;
            case GLFW_KEY_DOWN:
                return ImGuiKey_DownArrow;
            case GLFW_KEY_PAGE_UP:
                return ImGuiKey_PageUp;
            case GLFW_KEY_PAGE_DOWN:
                return ImGuiKey_PageDown;
            case GLFW_KEY_HOME:
                return ImGuiKey_Home;
            case GLFW_KEY_END:
                return ImGuiKey_End;
            case GLFW_KEY_INSERT:
                return ImGuiKey_Insert;
            case GLFW_KEY_DELETE:
                return ImGuiKey_Delete;
            case GLFW_KEY_BACKSPACE:
                return ImGuiKey_Backspace;
            case GLFW_KEY_SPACE:
                return ImGuiKey_Space;
            case GLFW_KEY_ENTER:
                return ImGuiKey_Enter;
            case GLFW_KEY_ESCAPE:
                return ImGuiKey_Escape;
            case GLFW_KEY_APOSTROPHE:
                return ImGuiKey_Apostrophe;
            case GLFW_KEY_COMMA:
                return ImGuiKey_Comma;
            case GLFW_KEY_MINUS:
                return ImGuiKey_Minus;
            case GLFW_KEY_PERIOD:
                return ImGuiKey_Period;
            case GLFW_KEY_SLASH:
                return ImGuiKey_Slash;
            case GLFW_KEY_SEMICOLON:
                return ImGuiKey_Semicolon;
            case GLFW_KEY_EQUAL:
                return ImGuiKey_Equal;
            case GLFW_KEY_LEFT_BRACKET:
                return ImGuiKey_LeftBracket;
            case GLFW_KEY_BACKSLASH:
                return ImGuiKey_Backslash;
            case GLFW_KEY_RIGHT_BRACKET:
                return ImGuiKey_RightBracket;
            case GLFW_KEY_GRAVE_ACCENT:
                return ImGuiKey_GraveAccent;
            case GLFW_KEY_CAPS_LOCK:
                return ImGuiKey_CapsLock;
            case GLFW_KEY_SCROLL_LOCK:
                return ImGuiKey_ScrollLock;
            case GLFW_KEY_NUM_LOCK:
                return ImGuiKey_NumLock;
            case GLFW_KEY_PRINT_SCREEN:
                return ImGuiKey_PrintScreen;
            case GLFW_KEY_PAUSE:
                return ImGuiKey_Pause;
            case GLFW_KEY_KP_0:
                return ImGuiKey_Keypad0;
            case GLFW_KEY_KP_1:
                return ImGuiKey_Keypad1;
            case GLFW_KEY_KP_2:
                return ImGuiKey_Keypad2;
            case GLFW_KEY_KP_3:
                return ImGuiKey_Keypad3;
            case GLFW_KEY_KP_4:
                return ImGuiKey_Keypad4;
            case GLFW_KEY_KP_5:
                return ImGuiKey_Keypad5;
            case GLFW_KEY_KP_6:
                return ImGuiKey_Keypad6;
            case GLFW_KEY_KP_7:
                return ImGuiKey_Keypad7;
            case GLFW_KEY_KP_8:
                return ImGuiKey_Keypad8;
            case GLFW_KEY_KP_9:
                return ImGuiKey_Keypad9;
            case GLFW_KEY_KP_DECIMAL:
                return ImGuiKey_KeypadDecimal;
            case GLFW_KEY_KP_DIVIDE:
                return ImGuiKey_KeypadDivide;
            case GLFW_KEY_KP_MULTIPLY:
                return ImGuiKey_KeypadMultiply;
            case GLFW_KEY_KP_SUBTRACT:
                return ImGuiKey_KeypadSubtract;
            case GLFW_KEY_KP_ADD:
                return ImGuiKey_KeypadAdd;
            case GLFW_KEY_KP_ENTER:
                return ImGuiKey_KeypadEnter;
            case GLFW_KEY_KP_EQUAL:
                return ImGuiKey_KeypadEqual;
            case GLFW_KEY_LEFT_SHIFT:
                return ImGuiKey_LeftShift;
            case GLFW_KEY_LEFT_CONTROL:
                return ImGuiKey_ModCtrl;
            case GLFW_KEY_LEFT_ALT:
                return ImGuiKey_LeftAlt;
            case GLFW_KEY_LEFT_SUPER:
                return ImGuiKey_LeftSuper;
            case GLFW_KEY_RIGHT_SHIFT:
                return ImGuiKey_RightShift;
            case GLFW_KEY_RIGHT_CONTROL:
                return ImGuiKey_RightCtrl;
            case GLFW_KEY_RIGHT_ALT:
                return ImGuiKey_RightAlt;
            case GLFW_KEY_RIGHT_SUPER:
                return ImGuiKey_RightSuper;
            case GLFW_KEY_MENU:
                return ImGuiKey_Menu;
            case GLFW_KEY_0:
                return ImGuiKey_0;
            case GLFW_KEY_1:
                return ImGuiKey_1;
            case GLFW_KEY_2:
                return ImGuiKey_2;
            case GLFW_KEY_3:
                return ImGuiKey_3;
            case GLFW_KEY_4:
                return ImGuiKey_4;
            case GLFW_KEY_5:
                return ImGuiKey_5;
            case GLFW_KEY_6:
                return ImGuiKey_6;
            case GLFW_KEY_7:
                return ImGuiKey_7;
            case GLFW_KEY_8:
                return ImGuiKey_8;
            case GLFW_KEY_9:
                return ImGuiKey_9;
            case GLFW_KEY_A:
                return ImGuiKey_A;
            case GLFW_KEY_B:
                return ImGuiKey_B;
            case GLFW_KEY_C:
                return ImGuiKey_C;
            case GLFW_KEY_D:
                return ImGuiKey_D;
            case GLFW_KEY_E:
                return ImGuiKey_E;
            case GLFW_KEY_F:
                return ImGuiKey_F;
            case GLFW_KEY_G:
                return ImGuiKey_G;
            case GLFW_KEY_H:
                return ImGuiKey_H;
            case GLFW_KEY_I:
                return ImGuiKey_I;
            case GLFW_KEY_J:
                return ImGuiKey_J;
            case GLFW_KEY_K:
                return ImGuiKey_K;
            case GLFW_KEY_L:
                return ImGuiKey_L;
            case GLFW_KEY_M:
                return ImGuiKey_M;
            case GLFW_KEY_N:
                return ImGuiKey_N;
            case GLFW_KEY_O:
                return ImGuiKey_O;
            case GLFW_KEY_P:
                return ImGuiKey_P;
            case GLFW_KEY_Q:
                return ImGuiKey_Q;
            case GLFW_KEY_R:
                return ImGuiKey_R;
            case GLFW_KEY_S:
                return ImGuiKey_S;
            case GLFW_KEY_T:
                return ImGuiKey_T;
            case GLFW_KEY_U:
                return ImGuiKey_U;
            case GLFW_KEY_V:
                return ImGuiKey_V;
            case GLFW_KEY_W:
                return ImGuiKey_W;
            case GLFW_KEY_X:
                return ImGuiKey_X;
            case GLFW_KEY_Y:
                return ImGuiKey_Y;
            case GLFW_KEY_Z:
                return ImGuiKey_Z;
            case GLFW_KEY_F1:
                return ImGuiKey_F1;
            case GLFW_KEY_F2:
                return ImGuiKey_F2;
            case GLFW_KEY_F3:
                return ImGuiKey_F3;
            case GLFW_KEY_F4:
                return ImGuiKey_F4;
            case GLFW_KEY_F5:
                return ImGuiKey_F5;
            case GLFW_KEY_F6:
                return ImGuiKey_F6;
            case GLFW_KEY_F7:
                return ImGuiKey_F7;
            case GLFW_KEY_F8:
                return ImGuiKey_F8;
            case GLFW_KEY_F9:
                return ImGuiKey_F9;
            case GLFW_KEY_F10:
                return ImGuiKey_F10;
            case GLFW_KEY_F11:
                return ImGuiKey_F11;
            case GLFW_KEY_F12:
                return ImGuiKey_F12;
            default:
                return ImGuiKey_None;
        }
    }

    void VulkanRenderer::mouseScroll(float yOffset) {
        mouseButtons.wheel += ((float) yOffset * mouseScrollSpeed);
        if (mouseButtons.wheel < -10.0f) {
            mouseButtons.wheel = -10.0f;
        }
        if (mouseButtons.wheel > 10.0f) {
            mouseButtons.wheel = 10.0f;
        }
    }


#ifdef WIN32
    void VulkanRenderer::clipboard() {
        // Try opening the clipboard
        if (! OpenClipboard(nullptr))
            return;

        // Get handle of clipboard object for ANSI text
        HANDLE hData = GetClipboardData(CF_TEXT);
        if (hData == nullptr)
            return;

        // Lock the handle to get the actual text pointer
        char * pszText = static_cast<char*>( GlobalLock(hData) );
        if (pszText == nullptr)
            return;

        // Save text in a string class instance
        glfwSetClipboardString(window, pszText);
        // Release the lock
        GlobalUnlock( hData );

        CloseClipboard();
    }
#endif
};
