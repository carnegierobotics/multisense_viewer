//
// Created by magnus on 8/28/21.
//


#include "VulkanRenderer.h"

VulkanRenderer::VulkanRenderer(const std::string &title, bool enableValidation) {
    settings.validation = enableValidation;

    // Create window instance
    // boilerplate stuff (ie. basic window setup, initialize OpenGL) occurs in abstract class
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(1280, 720, title.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, VulkanRenderer::keyCallback);
    glfwSetWindowSizeCallback(window, VulkanRenderer::resizeCallback);
    glfwSetMouseButtonCallback(window, VulkanRenderer::mouseButtonCallback);
    glfwSetCursorPosCallback(window, VulkanRenderer::cursorPositionCallback);
    glfwSetScrollCallback(window, VulkanRenderer::mouseScrollCallback);
    glfwSetCharCallback(window, VulkanRenderer::charCallback);

}

VkResult VulkanRenderer::createInstance(bool enableValidation) {
    settings.validation = enableValidation;
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = name.c_str();
    appInfo.pEngineName = name.c_str();

    auto FN_vkEnumerateInstanceVersion = PFN_vkEnumerateInstanceVersion(
            vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (&vkEnumerateInstanceVersion) {
        vkEnumerateInstanceVersion(&apiVersion);
    }
    appInfo.apiVersion = apiVersion;

    pLogger->info("Setting up vulkan with API Version: {}.{}.{} Minimum recommended version to use is 1.2.0",
                  VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion), VK_VERSION_PATCH(apiVersion));


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
    // If requested, we enable the default validation layers for debugging
// If requested, we enable the default validation layers for debugging
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
    // Get list of devices and capabilities of each device
    uint32_t gpuCount = 0;
    err = vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
    if (err != VK_SUCCESS or gpuCount == 0) {
        throw std::runtime_error("No device with vulkan support found");
    }
    // Enumerate devices
    std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
    err = vkEnumeratePhysicalDevices(instance, &gpuCount, physicalDevices.data());
    if (err != VK_SUCCESS) {
        throw std::runtime_error("Could not enumerate physical devices");
    }
    // Select physical device to be used for the Vulkan example
    // Defaults to the first device unless anything else specified

    physicalDevice = pickPhysicalDevice(physicalDevices);
    // Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

    VkPhysicalDeviceVulkan11Features features;
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features.pNext = nullptr;


    // Derived examples can override this to set actual features (based on above readings) to enable for logical device creation
    addDeviceFeatures();

    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features;
    features2.features = deviceFeatures;

    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    // If available then: Add KHR_SAMPLER_YCBCR For Color camera data format.
    if (features.samplerYcbcrConversion) {
        enabledDeviceExtensions.push_back(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
        pLogger->info("Enabling YCBCR Sampler Extension");
    }else {
        pLogger->error("YCBCR Sampler support not found!");
    }

    // Vulkan device creation
    // This is firstUpdate by a separate class that gets a logical device representation
    // and encapsulates functions related to a device
    vulkanDevice = new VulkanDevice(physicalDevice);
    err = vulkanDevice->createLogicalDevice(enabledFeatures, enabledDeviceExtensions, &features);
    if (err != VK_SUCCESS)
        throw std::runtime_error("Failed to create logical device");

    device = vulkanDevice->logicalDevice;
    // Get a graphics queue from the device
    vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.graphics, 0, &queue);
    // Find a suitable depth format
    depthFormat = Utils::findDepthFormat(physicalDevice);

    swapchain.connect(instance, physicalDevice, device);
    // Create synchronization objects
    VkSemaphoreCreateInfo semaphoreCreateInfo = Populate::semaphoreCreateInfo();
    // Create a semaphore used to synchronize image presentation
    // Ensures that the image is displayed before we start submitting new commands to the queue
    err = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores.presentComplete);
    if (err)
        throw std::runtime_error("Failed to create semaphore");
    // Create a semaphore used to synchronize command submission
    // Ensures that the image is not presented until all commands have been submitted and executed
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
    swapchain.cleanup();

    //vkDestroyCommandPool(device, cmdPool, nullptr);

    vkDestroySemaphore(device, semaphores.presentComplete, nullptr);
    vkDestroySemaphore(device, semaphores.renderComplete, nullptr);

    if (settings.validation)
        Validation::DestroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, nullptr);
    delete vulkanDevice;

    vkDestroyInstance(instance, nullptr);

    // CleanUp GLFW window
    glfwDestroyWindow(window);
    glfwTerminate();
}

void VulkanRenderer::addDeviceFeatures() {
}

void VulkanRenderer::viewChanged() {

}

void VulkanRenderer::keyPressed(uint32_t) {

}

void VulkanRenderer::mouseMoved(double x, double y, bool &handled) {

}

void VulkanRenderer::windowResized() {

}

void VulkanRenderer::buildCommandBuffers() {

}

void VulkanRenderer::setupDepthStencil() {
    VkImageCreateInfo imageCI{};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = depthFormat;
    imageCI.extent = {width, height, 1};
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    VkResult result = vkCreateImage(device, &imageCI, nullptr, &depthStencil.image);
    if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth image");

    VkMemoryRequirements memReqs{};
    vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);

    VkMemoryAllocateInfo memAllloc{};
    memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllloc.allocationSize = memReqs.size;
    memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    result = vkAllocateMemory(device, &memAllloc, nullptr, &depthStencil.mem);
    if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth image memory");
    result = vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0);
    if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth image memory");

    VkImageViewCreateInfo imageViewCI{};
    imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
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
    if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth image view");
}

void VulkanRenderer::setupFrameBuffer() {
    VkImageView attachments[2];
    // Depth/Stencil attachment is the same for all frame buffers
    attachments[1] = depthStencil.view;
    VkFramebufferCreateInfo frameBufferCreateInfo = {};
    frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferCreateInfo.pNext = NULL;
    frameBufferCreateInfo.renderPass = renderPass;
    frameBufferCreateInfo.attachmentCount = 2;
    frameBufferCreateInfo.pAttachments = attachments;
    frameBufferCreateInfo.width = width;
    frameBufferCreateInfo.height = height;
    frameBufferCreateInfo.layers = 1;

    // Create frame buffers for every swap chain image
    frameBuffers.resize(swapchain.imageCount);
    for (uint32_t i = 0; i < frameBuffers.size(); i++) {
        attachments[0] = swapchain.buffers[i].view;
        VkResult result = vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &frameBuffers[i]);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");
    }
}

void VulkanRenderer::setupRenderPass() {
    std::array<VkAttachmentDescription, 2> attachments = {};
    // Color attachment
    attachments[0].format = swapchain.colorFormat;
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
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

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

    VkResult result = (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
    if (result != VK_SUCCESS) throw std::runtime_error("Failed to create render pass");
}

void VulkanRenderer::createCommandPool() {
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = swapchain.queueNodeIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkResult result = vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool);
    if (result != VK_SUCCESS) throw std::runtime_error("Failed to create command pool");
}

void VulkanRenderer::createCommandBuffers() {
    // Create one command buffer for each swap chain image and reuse for rendering
    drawCmdBuffers.resize(swapchain.imageCount);

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


VkPipelineShaderStageCreateInfo VulkanRenderer::loadShader(const std::string &fileName, VkShaderStageFlagBits stage) {
    VkPipelineShaderStageCreateInfo shaderStage = {};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = stage;
    shaderStage.module = Utils::loadShader((Utils::getShadersPath() + fileName).c_str(), device);
    shaderStage.pName = "main";
    assert(shaderStage.module != VK_NULL_HANDLE);
    // TODO CLEANUP SHADERMODULES WHEN UNUSED AND ON EXITING VIEWER APP
    return shaderStage;
}


void VulkanRenderer::createPipelineCache() {
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkResult result = vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache);
    if (result != VK_SUCCESS) throw std::runtime_error("Failed to create command pool");
}


void VulkanRenderer::prepare() {
    swapchain.initSurface(window);
    createCommandPool();
    swapchain.create(&width, &height, settings.vsync);
    createCommandBuffers();
    createSynchronizationPrimitives();
    setupDepthStencil();
    setupRenderPass();
    createPipelineCache();
    setupFrameBuffer();

    guiManager = new AR::GuiManager(vulkanDevice);
    std::vector<VkPipelineShaderStageCreateInfo> shaders;
    shaders = {
            loadShader("imgui/ui.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader("imgui/ui.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
    };
    guiManager->setup((float) width, (float) height, renderPass, queue, &shaders);
    pLogger->info("Initialized GUI with shaders, ui.vert and ui.frag");

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
    resized = true;

    // Ensure all operations on the device have been finished before destroying resources
    vkQueueWaitIdle(queue);
    vkDeviceWaitIdle(device);

    // Recreate swap chain
    swapchain.create(&width, &height);

    // Recreate the frame buffers
    vkDestroyImageView(device, depthStencil.view, nullptr);
    vkDestroyImage(device, depthStencil.image, nullptr);
    vkFreeMemory(device, depthStencil.mem, nullptr);
    setupDepthStencil();
    for (uint32_t i = 0; i < frameBuffers.size(); i++) {
        vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
    }
    setupFrameBuffer();

    // Maybe resize overlay too
    // --- Missing code snippet ---

    // Command buffers need to be recreated as they may store
    // references to the recreated frame buffer
    destroyCommandBuffers();
    createCommandBuffers();
    buildCommandBuffers();
    vkDeviceWaitIdle(device);

    if ((width > 0.0f) && (height > 0.0f)) {
        camera.updateAspectRatio((float) width / (float) height);
    }
    pLogger->info("Window Resized. New size is: {} x {}", width, height);

    // Notify derived class
    windowResized();
    viewChanged();

    backendInitialized = true;
}


void VulkanRenderer::renderLoop() {
    destWidth = width;
    destHeight = height;

    auto graphLastTimestamp = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        auto tStart = std::chrono::high_resolution_clock::now();
        glfwPollEvents();
        if (viewUpdated) {
            viewUpdated = false;
            viewChanged();
        }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - startTime;
        runTime = elapsed_seconds.count();

        ImGuiIO &io = ImGui::GetIO();
        io.DisplaySize = ImVec2((float) width, (float) height);
        io.DeltaTime = frameTimer;
        io.WantCaptureMouse = true;
        io.MousePos = ImVec2(mousePos.x, mousePos.y);
        io.MouseWheel = mouseButtons.wheel;

        io.MouseDown[0] = mouseButtons.left;
        io.MouseDown[1] = mouseButtons.right;

        render();

        keypress = 0; // reset keypress
        mouseButtons.wheel = 0.0f; // Reset mousewheel after we rendered frames with imgui.

        auto tEnd = std::chrono::high_resolution_clock::now();

        frameCounter++;

        float fpsTimer = std::chrono::duration<double, std::milli>(tEnd - graphLastTimestamp).count();
        if (fpsTimer > 1000.0f) {
            lastFPS = (float) frameCounter * (1000.0f / fpsTimer);
            frameCounter = 0;
            graphLastTimestamp = tEnd;
        }

        auto tDiff = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - tStart).count();

        frameTimer = (float) tDiff / 1000.0f;

        // 60 FPS with with fpsTimeLimiter

        /*
        double fpsTimeLimiter = 0.01666666666;
        if (frameTimer < fpsTimeLimiter){
            double us = (fpsTimeLimiter - frameTimer) * 1000 * 1000;
            std::this_thread::sleep_for(std::chrono::microseconds (static_cast<long>(us)));
        }
        */

        camera.update(frameTimer);
        if (camera.moving()) {
            viewUpdated = true;
        }


    }
    // Flush device to make sure all resources can be freed
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
    }
}

void VulkanRenderer::prepareFrame() {
    // Acquire the next image from the swap chain
    VkResult result = swapchain.acquireNextImage(semaphores.presentComplete, &currentBuffer);
    // Recreate the swapchain if it's no longer compatible with the surface (OUT_OF_DATE) or no longer optimal for presentation (SUBOPTIMAL)
    if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
        windowResize();
    } else if (result != VK_SUCCESS)
        throw std::runtime_error("Failed to acquire next image");

    // Use a fence to wait until the command buffer has finished execution before using it again
    result = vkWaitForFences(device, 1, &waitFences[currentBuffer], VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS)
        throw std::runtime_error("Failed to wait for fence");
    result = vkResetFences(device, 1, &waitFences[currentBuffer]);
    if (result != VK_SUCCESS)
        throw std::runtime_error("Failed to reset fence");
}

void VulkanRenderer::submitFrame() {
    VkResult result = swapchain.queuePresent(queue, currentBuffer, semaphores.renderComplete);
    if (!((result == VK_SUCCESS) || (result == VK_SUBOPTIMAL_KHR))) {
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            // Swap chain is no longer compatible with the surface and needs to be recreated
            windowResize();
            return;
        }
    } else if (result != VK_SUCCESS)
        throw std::runtime_error("Failed to acquire next image");

    if (vkQueueWaitIdle(queue) != VK_SUCCESS)
        throw std::runtime_error("Failed to wait for Queue Idle");
}


void VulkanRenderer::UIUpdate(AR::GuiObjectHandles *uiSettings) {

}

void VulkanRenderer::updateOverlay() {
    // Update imGui

    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float) width, (float) height);
    io.DeltaTime = frameTimer;
    io.MousePos = ImVec2(mousePos.x, mousePos.y);

    io.MouseDown[0] = mouseButtons.left;
    io.MouseDown[1] = mouseButtons.right;

    guiManager->update((frameCounter == 0), frameTimer, width, height);
    if (guiManager->updateBuffers())
        buildCommandBuffers();

    UIUpdate(&guiManager->handles);

    /*
    UIOverlay->newFrame((frameCounter == 0), frameTimer, width, height);
    if (UIOverlay->updateBuffers()) {
        buildCommandBuffers();

    }

    if (UIOverlay->updated || UIOverlay->firstUpdate) {
        UIUpdate(UIOverlay->uiSettings);
        UIOverlay->firstUpdate = false;
    }
     */
}

void VulkanRenderer::renderFrame() {

}

/** CALLBACKS **/

void VulkanRenderer::setWindowSize(uint32_t width, uint32_t height) {
    destWidth = width;
    destHeight = height;
    windowResize();
}

void VulkanRenderer::resizeCallback(GLFWwindow *window, int width, int height) {
    auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
    myApp->setWindowSize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
}


void VulkanRenderer::charCallback(GLFWwindow *window, unsigned int codepoint) {
    auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));

    ImGuiIO &io = ImGui::GetIO();
    io.AddInputCharacter((unsigned short) codepoint);

}


void VulkanRenderer::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
    if ((key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) && action == GLFW_PRESS) {
        myApp->pLogger->info("Escape or Quit (Q) key registered. Closing program..");
        glfwSetWindowShouldClose(window, true);
    }

    ImGuiIO &io = ImGui::GetIO();
    io.AddKeyEvent(ImGuiKey_ModCtrl, (mods & GLFW_MOD_CONTROL) != 0);
    io.AddKeyEvent(ImGuiKey_ModShift, (mods & GLFW_MOD_SHIFT) != 0);
    io.AddKeyEvent(ImGuiKey_ModAlt, (mods & GLFW_MOD_ALT) != 0);
    io.AddKeyEvent(ImGuiKey_ModSuper, (mods & GLFW_MOD_SUPER) != 0);

    key = ImGui_ImplGlfw_TranslateUntranslatedKey(key, scancode);
    ImGuiKey imgui_key = ImGui_ImplGlfw_KeyToImGuiKey(key);
    io.AddKeyEvent(imgui_key, (action == GLFW_PRESS));


    // If we do not want to send key events to the Render engine but keem them in the UI.
    //if (io.WantCaptureKeyboard)
    //    return;
    myApp->keypress = key;

    if (action == GLFW_PRESS) {

        switch (key) {
            case GLFW_KEY_W:
                myApp->camera.keys.up = true;
                break;
            case GLFW_KEY_S:
                myApp->camera.keys.down = true;
                break;
            case GLFW_KEY_A:
                myApp->camera.keys.left = true;
                break;
            case GLFW_KEY_D:
                myApp->camera.keys.right = true;
            default:
                break;
        }
    }
    if (action == GLFW_RELEASE) {
        switch (key) {
            case GLFW_KEY_W:
                myApp->camera.keys.up = false;
                break;
            case GLFW_KEY_S:
                myApp->camera.keys.down = false;
                break;
            case GLFW_KEY_A:
                myApp->camera.keys.left = false;
                break;
            case GLFW_KEY_D:
                myApp->camera.keys.right = false;
            default:
                break;
        }
    }
}

void VulkanRenderer::handleMouseMove(int32_t x, int32_t y) {
    int32_t dx = (int32_t) mousePos.x - x;
    int32_t dy = (int32_t) mousePos.y - y;

    bool handled = false;

    if (settings.overlay) {
        ImGuiIO& io = ImGui::GetIO();
        io.WantCaptureMouse = true;
    }
    mouseMoved((float) x, (float) y, handled);

    if (handled) {
        mousePos = glm::vec2((float) x, (float) y);
        return;
    }

    if (mouseButtons.left) {
        camera.rotate(glm::vec3(dy * camera.rotationSpeed, -dx * camera.rotationSpeed, 0.0f));
        viewUpdated = true;
    }
    if (mouseButtons.right) {
        camera.translate(glm::vec3(-0.0f, 0.0f, dy * .005f));
        viewUpdated = true;
    }
    if (mouseButtons.middle) {
        camera.translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.0f));
        viewUpdated = true;
    }
    mousePos = glm::vec2((float) x, (float) y);
}

void VulkanRenderer::cursorPositionCallback(GLFWwindow *window, double xPos, double yPos) {
    auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
    myApp->handleMouseMove(xPos, yPos);

}

void VulkanRenderer::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS) {
        switch (button) {
            case GLFW_MOUSE_BUTTON_RIGHT:
                myApp->mouseButtons.right = true;
            case GLFW_MOUSE_BUTTON_MIDDLE:
                myApp->mouseButtons.middle = true;
            case GLFW_MOUSE_BUTTON_LEFT:
                myApp->mouseButtons.left = true;
        }
    }

    if (action == GLFW_RELEASE) {
        switch (button) {
            case GLFW_MOUSE_BUTTON_RIGHT:
                myApp->mouseButtons.right = false;
            case GLFW_MOUSE_BUTTON_MIDDLE:
                myApp->mouseButtons.middle = false;
            case GLFW_MOUSE_BUTTON_LEFT:
                myApp->mouseButtons.left = false;
        }
    }
}

void VulkanRenderer::mouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
    auto *myApp = static_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
    ImGuiIO &io = ImGui::GetIO();

    double scrollSpeed = 0.80f;
    myApp->mouseButtons.wheel += (float) (yoffset * scrollSpeed);

}

VkPhysicalDevice VulkanRenderer::pickPhysicalDevice(std::vector<VkPhysicalDevice> devices) {

    if (devices.empty())
        throw std::runtime_error("No physical devices available");

    for (auto &device: devices) {
        VkPhysicalDeviceProperties properties{};
        VkPhysicalDeviceFeatures features{};
        VkPhysicalDeviceMemoryProperties memoryProperties{};

        vkGetPhysicalDeviceProperties(device, &properties);
        vkGetPhysicalDeviceFeatures(device, &features);
        vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties);

        pLogger->info("Found physical device: {}, ", properties.deviceName);

        // Search for a discrete GPU and prefer this one
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            pLogger->info("Picked Discrete GPU. Name: {}, ", properties.deviceName);
            return device;
        }

    }

    // If no discrete GPU were found just return the first device found
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
    // if (action == GLFW_PRESS) printf("key %d scancode %d name '%s'\n", key, scancode, key_name);
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
            return ImGuiKey_LeftCtrl;
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


