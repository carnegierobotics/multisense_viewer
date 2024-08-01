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
#include <vk_mem_alloc.h>

#include "Viewer/VkRender/Core/VulkanRenderer.h"

#include "Viewer/Tools/Populate.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/VkRender/Core/VulkanResourceManager.h"

#ifndef VKRENDER_MULTISENSE_VIEWER_PRODUCTION

#include "Viewer/VkRender/Core/Validation.h"

#endif

namespace VkRender {
    VulkanRenderer::VulkanRenderer(const std::string &title) {
#ifdef VKRENDER_MULTISENSE_VIEWER_PRODUCTION
        m_settings.validation = false;
#else
        m_settings.validation = true;
#endif

        m_width = RendererConfig::getInstance().getUserSetting().applicationWidth;
        m_height = RendererConfig::getInstance().getUserSetting().applicationHeight;


        // Create window instance
        // boilerplate stuff (ie. basic window setup, initialize OpenGL) occurs in abstract class
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        //m_Width = 1200;
        //m_Height = 540;
        window = glfwCreateWindow(static_cast<int>(m_width), static_cast<int>(m_height), title.c_str(), nullptr,
                                  nullptr);
        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, this);
        glfwSetKeyCallback(window, VulkanRenderer::keyCallback);
        glfwSetWindowSizeCallback(window, VulkanRenderer::resizeCallback);
        glfwSetMouseButtonCallback(window, VulkanRenderer::mouseButtonCallback);
        glfwSetCursorPosCallback(window, VulkanRenderer::cursorPositionCallback);
        glfwSetScrollCallback(window, VulkanRenderer::mouseScrollCallback);
        glfwSetCharCallback(window, VulkanRenderer::charCallback);
        glfwSetWindowSizeLimits(window, m_width / 4, m_height / 4, GLFW_DONT_CARE, GLFW_DONT_CARE);
        m_cursors.resizeVertical = glfwCreateStandardCursor(GLFW_RESIZE_NS_CURSOR);
        m_cursors.resizeHorizontal = glfwCreateStandardCursor(GLFW_RESIZE_EW_CURSOR);
        m_cursors.crossHair = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
        m_cursors.arrow = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
        m_cursors.hand = glfwCreateStandardCursor(GLFW_POINTING_HAND_CURSOR);

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
        m_settings.validation = enableValidation;
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = m_name.c_str();
        appInfo.pEngineName = m_name.c_str();
        appInfo.apiVersion = apiVersion;
        m_logger->info("Setting up vulkan with API Version: {}.{}.{}",
                       VK_API_VERSION_MAJOR(apiVersion), VK_API_VERSION_MINOR(apiVersion),
                       VK_API_VERSION_PATCH(apiVersion));
        // Get extensions supported by the instance
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (m_settings.validation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        enabledInstanceExtensions = extensions;


        std::vector<const char *> instanceExtensions = {
                VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
        };
        enabledInstanceExtensions.insert(enabledInstanceExtensions.end(), instanceExtensions.begin(),
                                         instanceExtensions.end());

        // Check if extensions are supported
        if (!checkInstanceExtensionSupport(enabledInstanceExtensions))
            throw std::runtime_error("Instance Extensions not supported");

        VkInstanceCreateInfo instanceCreateInfo = {};
        instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.pNext = nullptr;
        instanceCreateInfo.pApplicationInfo = &appInfo;
        if (!enabledInstanceExtensions.empty()) {
            instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledInstanceExtensions.size());
            instanceCreateInfo.ppEnabledExtensionNames = enabledInstanceExtensions.data();
        }
        const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
        // The VK_LAYER_KHRONOS_validation contains all current validation functionality.
#ifdef VKRENDER_MULTISENSE_VIEWER_DEBUG
        if (m_settings.validation) {
            // Check if this layer is available at instance level
            if (Validation::checkValidationLayerSupport(validationLayers)) {
                instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
                instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                m_logger->info("Enabling Validation Layers");
            } else {
                std::cerr << "Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled\n";
                m_logger->info("Disabled Validation Layers since it was not found");
            }
        }
#endif
        return vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
    }

    bool VulkanRenderer::initVulkan() {
        // Create Instance
        VkResult err = createInstance(m_settings.validation);
        if (err) {
            throw std::runtime_error("Could not create Vulkan instance");
        }
        m_logger->info("Vulkan Instance successfully created");
        // If requested, we flashing the default validation layers for debugging
        // If requested, we flashing the default validation layers for debugging
#ifdef VKRENDER_MULTISENSE_VIEWER_DEBUG

        if (m_settings.validation) {
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
#endif
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

        msaaSamples = getMaxUsableSampleCount();
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
            //enabledDeviceExtensions.push_back(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
            //VkRender::RendererConfig::getInstance().addEnabledExtension(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
        } else {
            m_logger->error("YCBCR Sampler Extension support not found!");
        }

        // Vulkan m_Device creation
        // This is firstUpdate by a separate class that gets a logical m_Device representation
        // and encapsulates functions related to a m_Device
        m_vulkanDevice = new VulkanDevice(physicalDevice, &queueSubmitMutex);
        err = m_vulkanDevice->createLogicalDevice(enabledFeatures, enabledDeviceExtensions, &features);
        if (err != VK_SUCCESS)
            throw std::runtime_error("Failed to create logical device");

        device = m_vulkanDevice->m_LogicalDevice;
        // Get a graphics queue from the m_Device
        vkGetDeviceQueue(device, m_vulkanDevice->m_QueueFamilyIndices.graphics, 0, &graphicsQueue);
        // Get Compute queue
        vkGetDeviceQueue(device, m_vulkanDevice->m_QueueFamilyIndices.compute, 0, &computeQueue);
        // Find a suitable depth m_Format
        depthFormat = Utils::findDepthFormat(physicalDevice);

        // Initialize vulkan memory allocator (VMA)
        VmaVulkanFunctions vulkanFunctions = {};
        vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
        vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_1;
        allocatorCreateInfo.physicalDevice = physicalDevice;
        allocatorCreateInfo.device = device;
        allocatorCreateInfo.instance = instance;
        allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

        allocatorCreateInfo.flags =
                VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT | VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;

        VkResult result = vmaCreateAllocator(&allocatorCreateInfo, &m_allocator);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create VMA allocator");
        }
        return true;
    }

    VulkanRenderer::~VulkanRenderer() {
        VulkanResourceManager::destroyInstance();

        vkDestroyImageView(device, m_depthStencil.view, nullptr);
        vmaDestroyImage(m_allocator, m_depthStencil.image, m_depthStencil.allocation);

        vkDestroyImageView(device, m_colorImage.view, nullptr);
        vmaDestroyImage(m_allocator, m_colorImage.image, m_colorImage.allocation);

        // CleanUP all vulkan resources
        swapchain->cleanup();

        vkDestroyCommandPool(device, cmdPool, nullptr);
        vkDestroyCommandPool(device, cmdPoolCompute, nullptr);
        for (auto &fence: waitFences) {
            vkDestroyFence(device, fence, nullptr);
        }
        for (auto &fence: computeInFlightFences) {
            vkDestroyFence(device, fence, nullptr);
        }

        vkDestroyPipelineCache(device, pipelineCache, nullptr);
        for (auto &semaphore: semaphores) {
            vkDestroySemaphore(device, semaphore.presentComplete, nullptr);
            vkDestroySemaphore(device, semaphore.renderComplete, nullptr);
            vkDestroySemaphore(device, semaphore.computeComplete, nullptr);
        }

#ifdef VKRENDER_MULTISENSE_VIEWER_DEBUG
        if (m_settings.validation)
            Validation::DestroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, nullptr);
        // Write to file
        char *statsString;
        vmaBuildStatsString(m_allocator, &statsString, VK_TRUE);
        std::string filePath = "./vma_stats.json";
        std::ofstream file(filePath);
        if (!file) {
            Log::Logger::getInstance()->error("Failed to open file for writing VMA stats: {}", filePath);
            vmaFreeStatsString(m_allocator, statsString);
        } else {
            file << statsString;
            file.close();
        }
        vmaFreeStatsString(m_allocator, statsString);
#endif
        vmaDestroyAllocator(m_allocator);

        delete m_vulkanDevice; //Call to destructor for smart pointer destroy logical m_Device before instance
        vkDestroyInstance(instance, nullptr);
        // Cleanup GLFW Resources
        glfwDestroyCursor(m_cursors.arrow);
        glfwDestroyCursor(m_cursors.hand);
        glfwDestroyCursor(m_cursors.resizeVertical);
        glfwDestroyCursor(m_cursors.resizeHorizontal);
        glfwDestroyCursor(m_cursors.crossHair);
        // CleanUp GLFW window
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void VulkanRenderer::addDeviceFeatures() {
    }


    void VulkanRenderer::windowResized(int32_t i, int32_t i1, double d, double d1) {
    }

    void VulkanRenderer::buildCommandBuffers() {
    }


    void VulkanRenderer::createCommandPool() {
        VkCommandPoolCreateInfo cmdPoolInfo = Populate::commandPoolCreateInfo();
        cmdPoolInfo.queueFamilyIndex = swapchain->queueNodeIndex;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkResult result = vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create command pool");

        VkCommandPoolCreateInfo cmdPoolComputeInfo = Populate::commandPoolCreateInfo();
        cmdPoolComputeInfo.queueFamilyIndex = m_vulkanDevice->m_QueueFamilyIndices.compute;
        cmdPoolComputeInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        result = vkCreateCommandPool(device, &cmdPoolComputeInfo, nullptr, &cmdPoolCompute);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create compute command pool");
    }

    void VulkanRenderer::createCommandBuffers() {
        // Create one command buffer for each swap chain m_Image and reuse for rendering
        drawCmdBuffers.buffers.resize(swapchain->imageCount);

        VkCommandBufferAllocateInfo cmdBufAllocateInfo =
                Populate::commandBufferAllocateInfo(
                        cmdPool,
                        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                        static_cast<uint32_t>(drawCmdBuffers.buffers.size()));

        VkResult result = vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, drawCmdBuffers.buffers.data());
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate command buffers");


        // Create one command buffer for each swap chain m_Image and reuse for rendering
        computeCommand.buffers.resize(swapchain->imageCount);

        VkCommandBufferAllocateInfo cmdBufAllocateComputeInfo = Populate::commandBufferAllocateInfo(
                cmdPoolCompute,
                VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                static_cast<uint32_t>(computeCommand.buffers.size()));

        result = vkAllocateCommandBuffers(device, &cmdBufAllocateComputeInfo, computeCommand.buffers.data());
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate command buffers");
    }

    void VulkanRenderer::createSynchronizationPrimitives() {
        // Create synchronization Objects
        // Create a semaphore used to synchronize m_Image presentation
        // Ensures that the m_Image is displayed before we start submitting new commands to the queue
        // Set up submit info structure
        // Semaphores will stay the same during application lifetime
        // Command buffer submission info is set by each example
        VkSemaphoreCreateInfo semaphoreCreateInfo = Populate::semaphoreCreateInfo();
        VkFenceCreateInfo fenceCreateInfo = Populate::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
        waitFences.resize(swapchain->imageCount);
        semaphores.resize(swapchain->imageCount);
        computeInFlightFences.resize(swapchain->imageCount);
        for (size_t i = 0; i < swapchain->imageCount; ++i) {
            if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores[i].presentComplete) !=
                VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores[i].renderComplete) != VK_SUCCESS ||
                vkCreateFence(device, &fenceCreateInfo, nullptr, &waitFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create graphics synchronization objects for a frame!");
            }
            if (vkCreateFence(device, &fenceCreateInfo, nullptr, &computeInFlightFences[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores[i].computeComplete) != VK_SUCCESS)
                throw std::runtime_error("Failed to create compute synchronization fence");
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
        info.vsync = m_settings.vsync;
        swapchain = std::make_unique<VulkanSwapchain>(info, &m_width, &m_height);

        createCommandPool();
        createCommandBuffers();
        createSynchronizationPrimitives();

        createPipelineCache();

        m_logger->info("Initialized Renderer backend");

        rendererStartTime = std::chrono::system_clock::now();
    }


    void VulkanRenderer::destroyCommandBuffers() {
        vkFreeCommandBuffers(device, cmdPool, static_cast<uint32_t>(drawCmdBuffers.buffers.size()),
                             drawCmdBuffers.buffers.data());
        vkFreeCommandBuffers(device, cmdPoolCompute, static_cast<uint32_t>(computeCommand.buffers.size()),
                             computeCommand.buffers.data());
    }

    void VulkanRenderer::windowResize() {
        if (!backendInitialized) {
            return;
        }
        backendInitialized = false;
        int32_t prevWidth = static_cast<int32_t>(m_width);
        int32_t prevHeight = static_cast<int32_t>(m_height);
        glfwGetFramebufferSize(window, reinterpret_cast<int *>(&m_width), reinterpret_cast<int *>(&m_height));
        int32_t widthChanged = static_cast<int32_t>(m_width) - prevWidth;
        int32_t heightChanged = static_cast<int32_t>(m_height) - prevHeight;
        double widthScale = static_cast<double>(m_width) / static_cast<double>(prevWidth);
        double heightScale = static_cast<double>(m_height) / static_cast<double>(prevHeight);

        // Suspend application while it is in minimized state
        // Also unsignal semaphore for presentation because we are recreating the swapchain
        while (m_width == 0 || m_height == 0) {
            glfwGetFramebufferSize(window, reinterpret_cast<int *>(&m_width), reinterpret_cast<int *>(&m_height));
            glfwWaitEvents();
        }
        // Ensure all operations on the m_Device have been finished before destroying resources
        vkQueueWaitIdle(graphicsQueue);
        vkDeviceWaitIdle(device);

        // Recreate swap chain
        swapchain->create(&m_width, &m_height, m_settings.vsync);

        VkSemaphoreCreateInfo semaphoreCreateInfo = Populate::semaphoreCreateInfo();
        // Create a semaphore used to synchronize m_Image presentation
        // Ensures that the m_Image is displayed before we start submitting new commands to the queue
        for (auto &semaphore: semaphores) {
            vkDestroySemaphore(device, semaphore.presentComplete, nullptr);
            VkResult err = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphore.presentComplete);
            if (err != VK_SUCCESS)
                throw std::runtime_error("Failed to create semaphore");
        }

        m_logger->info("Window Resized. New size is: {} x {}", m_width, m_height);

        // Notify derived class
        windowResized(widthChanged, heightChanged, widthScale, heightScale);

        // Command buffers need to be recreated as they may store
        // references to the recreated frame buffer
        destroyCommandBuffers();
        createCommandBuffers();
        buildCommandBuffers();
        vkDeviceWaitIdle(device);

        backendInitialized = true;
    }

    void VulkanRenderer::renderLoop() {
        destWidth = m_width;
        destHeight = m_height;
        auto graphLastTimestamp = std::chrono::high_resolution_clock::now();
        drawCmdBuffers.frameIndex = &currentFrame;
        drawCmdBuffers.activeImageIndex = &imageIndex;

        while (!glfwWindowShouldClose(window)) {
            auto tStart = std::chrono::high_resolution_clock::now();
            frameID++; // First frame will have id 1.
            glfwPollEvents();
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<float, std::milli> elapsed_milliseconds = end - rendererStartTime;
            runTime = elapsed_milliseconds.count();
            Log::Logger::getInstance()->trace("Mouse events are: ({},{})  action: {}", mouse.x, mouse.y, mouse.action);
            /** Compute pipeline command recording and submission **/
            //computePipeline(); // TODO Either implement or remove
            updateUniformBuffers();
            /** Aquire next image **/
            prepareFrame();
            /** Call Renderer's render function **/
            recordCommands();
            /** Present frame **/
            submitFrame();

            /** FrameTiming **/
            auto tEnd = std::chrono::high_resolution_clock::now();
            frameCounter++;
            float fpsTimer = std::chrono::duration<float, std::milli>(tEnd - graphLastTimestamp).count();
            if (fpsTimer > 333.0f) {
                lastFPS = static_cast<float>(frameCounter) * (1000.0f / fpsTimer);
                frameCounter = 0;
                graphLastTimestamp = tEnd;
            }
            auto tDiff = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - tStart).count();
            frameTimer = static_cast<float>(tDiff) / 1000.0f;

            postRenderActions();
            mouse.d = glm::vec2(0.0f);
            mouse.action = -1;
            Log::Logger::getInstance()->trace("Reset Mouse action to -1");
        }
        // Flush m_Device to make sure all resources can be freed before we start cleanup
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
    }


    void VulkanRenderer::computePipeline() {
        VkSubmitInfo sInfo{};
        sInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        // Compute submission
        // Only wait on the fence if we submitted work last time
        if (vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX != VK_SUCCESS))
            throw std::runtime_error("Failed to wait for compute fence");


        vkResetFences(device, 1, &computeInFlightFences[currentFrame]);

        vkResetCommandBuffer(computeCommand.buffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        /** call renderer compute function **/
        compute();
        sInfo.commandBufferCount = 1;
        sInfo.pCommandBuffers = &computeCommand.buffers[currentFrame];
        sInfo.signalSemaphoreCount = 1;
        sInfo.pSignalSemaphores = &semaphores[currentFrame].computeComplete;
        if (vkQueueSubmit(computeQueue, 1, &sInfo, computeInFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit compute command buffer!");
        }
    }


    void VulkanRenderer::prepareFrame() {
        // Use a fence to wait until the command buffer has finished execution before using it again
        if (vkWaitForFences(device, 1, &waitFences[currentFrame], VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            throw std::runtime_error("Failed to wait for render fence");

        // Cleanup deferred deletions
        VulkanResourceManager::getInstance().cleanup();


        if (recreateResourcesNextFrame) {
            Log::Logger::getInstance()->info("Attempting to launch resize window to solve vkSubmit Issue");
            windowResize();
            recreateResourcesNextFrame = false;
        }
        // Acquire the next m_Image from the swap chain
        VkResult result = swapchain->acquireNextImage(semaphores[currentFrame].presentComplete, &imageIndex);
        // Recreate the swapchain if it's no longer compatible with the surface (OUT_OF_DATE) or no longer optimal for presentation (SUBOPTIMAL)
        if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
            Log::Logger::getInstance()->info("SwapChain no longer compatible on acquire next image. Recreating..");
            windowResize();
            // New swapchain so we dont need to wait for present complete sempahore
            VkResult res = swapchain->acquireNextImage(semaphores[currentFrame].presentComplete, &imageIndex);
            if (res != VK_SUCCESS) {
                throw std::runtime_error(
                        "Failed to acquire next m_Image in prepareFrame after windowresize. VkResult: " +
                        std::to_string(result));
            }
        } else if (result != VK_SUCCESS)
            throw std::runtime_error(
                    "Failed to acquire next m_Image in prepareFrame. VkResult: " + std::to_string(result));


        result = vkResetFences(device, 1, &waitFences[currentFrame]);
        if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to reset fence");

        vkResetCommandBuffer(drawCmdBuffers.buffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
    }

    void VulkanRenderer::submitFrame() {

        std::unique_lock<std::mutex> lock(queueSubmitMutex);
        VkSemaphore waitSemaphores[] = {
                //semaphores[currentFrame].computeComplete,
                semaphores[currentFrame].presentComplete,
                //updateVulkan
        };
        VkPipelineStageFlags waitStages[] = {
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        };
        VkSemaphore signalSemaphores[] = {
                semaphores[currentFrame].renderComplete,
        };

        submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &drawCmdBuffers.buffers[currentFrame];
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, waitFences[currentFrame]);

        VkResult result = swapchain->queuePresent(graphicsQueue, imageIndex,
                                                  semaphores[currentFrame].renderComplete);
        if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR) {
            // Swap chain is no longer compatible with the surface and needs to be recreated
            Log::Logger::getInstance()->warning(
                    "SwapChain no longer compatible on graphicsQueue present. Will recreate on next frame");
            recreateResourcesNextFrame = true;
        } else if (result != VK_SUCCESS) {
            Log::Logger::getInstance()->error("Suboptimal Surface: Failed to acquire next m_Image. VkResult: {}",
                                              std::to_string(result));
        }


        currentFrame = (currentFrame + 1) % swapchain->imageCount;
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
            if (myApp->destWidth != static_cast<uint32_t>(width) &&
                myApp->destHeight != static_cast<uint32_t>(height))
                myApp->setWindowSize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
        }
    }

    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

    void VulkanRenderer::charCallback(GLFWwindow *window, unsigned int codepoint) {
        /*
        ImGuiIO &io = ImGui::GetIO();
        io.AddInputCharacter(static_cast<unsigned short>(codepoint));
    */
    }

    DISABLE_WARNING_POP


    void VulkanRenderer::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
        auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
        if ((key == GLFW_KEY_ESCAPE) && action == GLFW_PRESS) {
            myApp->m_logger->info("Escape key registered");
            myApp->closeApplication();
        }
        /*
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

        */

#ifdef WIN32
        if ((mods & GLFW_MOD_CONTROL) != 0 && key == GLFW_KEY_V) {
            myApp->clipboard();
        }
#endif

        if (action == GLFW_PRESS) {
            switch (key) {
                case GLFW_KEY_W:
                case GLFW_KEY_UP:
                    myApp->input.keys.up = true;
                    break;
                case GLFW_KEY_S:
                case GLFW_KEY_DOWN:
                    myApp->input.keys.down = true;
                    break;
                case GLFW_KEY_A:
                case GLFW_KEY_LEFT:
                    myApp->input.keys.left = true;
                    break;
                case GLFW_KEY_D:
                case GLFW_KEY_RIGHT:
                    myApp->input.keys.right = true;
                default:
                    break;
            }
        }
        if (action == GLFW_RELEASE) {
            switch (key) {
                case GLFW_KEY_W:
                case GLFW_KEY_UP:
                    myApp->input.keys.up = false;
                    break;
                case GLFW_KEY_S:
                case GLFW_KEY_DOWN:
                    myApp->input.keys.down = false;
                    break;
                case GLFW_KEY_A:
                case GLFW_KEY_LEFT:
                    myApp->input.keys.left = false;
                    break;
                case GLFW_KEY_D:
                case GLFW_KEY_RIGHT:
                    myApp->input.keys.right = false;
                default:
                    break;
            }
        }

        myApp->keyboardCallback(window, key, scancode, action, mods);
    }

    void VulkanRenderer::handleMouseMove(float x, float y) {
        bool handled = false;

        /*
        if (m_settings.overlay) {
            ImGuiIO &io = ImGui::GetIO();
            io.WantCaptureMouse = true;
        }
        */

        mouseMoved(x, y, handled);
    }

    void VulkanRenderer::cursorPositionCallback(GLFWwindow *window, double xPos, double yPos) {
        auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
        myApp->handleMouseMove(static_cast<float>(xPos), static_cast<float>(yPos));
    }

    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

    void VulkanRenderer::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
        auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));

        if (action == GLFW_PRESS) {
            switch (button) {
                case GLFW_MOUSE_BUTTON_RIGHT:
                    myApp->mouse.right = true;
                    break;
                case GLFW_MOUSE_BUTTON_MIDDLE:
                    myApp->mouse.middle = true;
                    break;
                case GLFW_MOUSE_BUTTON_LEFT:
                    myApp->mouse.left = true;
                    break;
                default:
                    break;
            }
        }
        if (action == GLFW_RELEASE) {
            switch (button) {
                case GLFW_MOUSE_BUTTON_RIGHT:
                    myApp->mouse.right = false;
                    break;
                case GLFW_MOUSE_BUTTON_MIDDLE:
                    myApp->mouse.middle = false;
                    break;
                case GLFW_MOUSE_BUTTON_LEFT:
                    myApp->mouse.left = false;
                    break;
                default:
                    break;
            }
        }
        myApp->mouse.action = action;

    }

    void VulkanRenderer::mouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
        auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
        /*
        ImGuiIO &io = ImGui::GetIO();
        myApp->mouseScroll(static_cast<float>(yoffset));
        io.MouseWheel += 0.5f * static_cast<float>(yoffset);
    */

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
            m_logger->info("Found physical d: {}, ", properties.deviceName);

            // Search for a discrete GPU and prefer this one
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                m_logger->info("Picked Discrete GPU. Name: {}, ", properties.deviceName);
                return d;
            }
        }
        // If no discrete GPU were found just return the first m_Device found
        return devices[0];
    }

    VkSampleCountFlagBits VulkanRenderer::getMaxUsableSampleCount() {
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

        VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts &
                                    physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
        if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
        if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
        if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
        if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
        if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

        return VK_SAMPLE_COUNT_1_BIT;
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
        mouse.wheel += yOffset * mouseScrollSpeed;
        if (mouse.wheel < -10.0f) {
            mouse.wheel = -10.0f;
        }
        if (mouse.wheel > 10.0f) {
            mouse.wheel = 10.0f;
        }
    }

    /*
    void VulkanRenderer::createColorResources() {
        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = swapchain->colorFormat;
        imageCI.extent = {m_width, m_height, 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = msaaSamples;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkResult result = vkCreateImage(device, &imageCI, nullptr, &m_colorImage.image);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image");

        VkMemoryRequirements memReqs{};
        vkGetImageMemoryRequirements(device, m_colorImage.image, &memReqs);

        VkMemoryAllocateInfo memAllloc = Populate::memoryAllocateInfo();
        memAllloc.allocationSize = memReqs.size;
        memAllloc.memoryTypeIndex = m_vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        result = vkAllocateMemory(device, &memAllloc, nullptr, &m_colorImage.mem);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth m_Image memory");
        result = vkBindImageMemory(device, m_colorImage.image, m_colorImage.mem, 0);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth m_Image memory");

        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.image = m_colorImage.image;
        imageViewCI.format = swapchain->colorFormat;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        result = vkCreateImageView(device, &imageViewCI, nullptr, &m_colorImage.view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image m_View");
    }

     */

#ifdef WIN32

    void VulkanRenderer::clipboard() {
        // Try opening the clipboard
        if (!OpenClipboard(nullptr))
            return;

        // Get handle of clipboard object for ANSI text
        HANDLE hData = GetClipboardData(CF_TEXT);
        if (hData == nullptr)
            return;

        // Lock the handle to get the actual text pointer
        char* pszText = static_cast<char*>(GlobalLock(hData));
        if (pszText == nullptr)
            return;
        // Save text in a string class instance
        glfwSetClipboardString(window, pszText);
        // Release the lock
        GlobalUnlock(hData);
        CloseClipboard();
    }
#endif

    bool VulkanRenderer::checkInstanceExtensionSupport(const std::vector<const char *> &checkExtensions) {
        //Need to get number of extensions to create array of correct size to hold extensions.
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        //Create a list of VkExtensionProperties using count.
        std::vector<VkExtensionProperties> extensions(extensionCount);
        //Populate the list of specific/named extensions
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        //Check if given extensions are in list of available extensions
        for (const auto &checkExtension: checkExtensions) {
            bool hasExtensions = false;
            for (const auto &extension: extensions) {
                if (strcmp(checkExtension, extension.extensionName) == 0) {
                    hasExtensions = true;
                    break;
                }
            }
            if (!hasExtensions) {
                return false;
            }
        }
        return true;
    }

    void VulkanRenderer::closeApplication() {
        m_logger->info("Closing application...");
        glfwSetWindowShouldClose(window, true);
    }

}
