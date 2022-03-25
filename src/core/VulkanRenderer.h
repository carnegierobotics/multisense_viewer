//
// Created by magnus on 8/28/21.
//

#ifndef MULTISENSE_VULKANRENDERER_H
#define MULTISENSE_VULKANRENDERER_H

#ifdef WIN32
    #ifdef WIN_DEBUG
        #pragma comment(linker, "/SUBSYSTEM:CONSOLE")
    #endif
#endif

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <string>
#include <vector>
#include "Camera.h"
#include <vector>
#include <cstring>
#include <iostream>
#include <glm/vec2.hpp>
#include <chrono>

#include <MultiSense/src/imgui/UISettings.h>
#include <MultiSense/src/imgui/VkGUI.h>
#include "VulkanSwapchain.h"
#include "Validation.h"
#include "VulkanDevice.h"


class VulkanRenderer {

public:
    VulkanRenderer(const std::string &title, bool enableValidation = false);

    virtual ~VulkanRenderer();

    /** @brief Setup the vulkan instance, enable required extensions and connect to the physical device (GPU) */
    bool initVulkan();

    /** @brief Example settings that can be changed by ... */
    struct Settings {
        /** @brief Activates validation layers (and message output) when set to true */
        bool validation = false;
        /** @brief Set to true if fullscreen mode has been requested via command line */
        bool fullscreen = false;
        /** @brief Set to true if v-sync will be forced for the swapchain */
        bool vsync = false;
        /** @brief Enable UI overlay */
        bool overlay = true;
    } settings;

    std::string title = "-1";
    std::string name = "VulkanRenderer";
    uint32_t apiVersion = VK_API_VERSION_1_3;
    bool backendInitialized = false;
    bool resized = false;
    uint32_t width = 1280;
    uint32_t height = 720;

    /** @brief Encapsulated physical and logical vulkan device */
    VulkanDevice *vulkanDevice{};

    struct {
        VkImage image;
        VkDeviceMemory mem;
        VkImageView view;
    } depthStencil{};

    /** @brief Last frame time measured using a high performance timer (if available) */
    float frameTimer = 1.0f;
    std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<double>> startTime;
    float runTime = 0.0f;

    Camera camera;
    glm::vec2 mousePos{};

    struct {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouseButtons;

    /** @brief Handle for UI updates and overlay */
    ImGUI* UIOverlay;

    /** @brief (Virtual) Creates the application wide Vulkan instance */
    virtual VkResult createInstance(bool enableValidation);
    /** @brief (Pure virtual) Render function to be implemented by the sample application */
    virtual void render() = 0;
    /** @brief (Virtual) Called when the camera view has changed */
    virtual void viewChanged();
    /** @brief (Virtual) Called after a key was pressed, can be used to do custom key handling */
    virtual void keyPressed(uint32_t);
    /** @brief (Virtual) Called once a update on the UI is detected */
    virtual void UIUpdate(UISettings *uiSettings);
    /** @brief (Virtual) Called after the mouse cursor moved and before internal events (like camera rotation) is firstUpdate */
    virtual void mouseMoved(double x, double y, bool &handled);
    /** @brief (Virtual) Called when the window has been resized, can be used by the sample application to recreate resources */
    virtual void windowResized();
    /** @brief (Virtual) Called when resources have been recreated that require a rebuild of the command buffers (e.g. frame buffer), to be implemented by the sample application */
    virtual void buildCommandBuffers();
    /** @brief (Virtual) Setup default depth and stencil views */
    virtual void setupDepthStencil();
    /** @brief (Virtual) Setup default framebuffers for all requested swapchain images */
    virtual void setupFrameBuffer();
    /** @brief (Virtual) Setup a default renderpass */
    virtual void setupRenderPass();
    /** @brief (Virtual) Called after the physical device features have been read, can be used to set features to enable on the device */
    virtual void addDeviceFeatures() = 0;
    /** @brief Prepares all Vulkan resources and functions required to run the sample */
    virtual void prepare();
    /** @brief Entry point for the main render loop */
    void renderLoop();
    /** @brief Adds the drawing commands for the ImGui overlay to the given command buffer */
    void drawUI(const VkCommandBuffer commandBuffer);
    /** Prepare the next frame for workload submission by acquiring the next swap chain image */
    void prepareFrame();
    /** @brief Presents the current image to the swap chain */
    void submitFrame();
    /** @brief (Virtual) Default image acquire + submission and command buffer submission function */
    virtual void renderFrame();


    VkPipelineShaderStageCreateInfo loadShader(const std::string& fileName, VkShaderStageFlagBits stage);

protected:
    // Window instance GLFW
    GLFWwindow* window;
    // Vulkan Instance, stores al per-application states
    VkInstance instance{};
    // Physical Device that Vulkan will use
    VkPhysicalDevice physicalDevice{};
    //Physical device properties (device limits etc..)
    VkPhysicalDeviceProperties deviceProperties{};
    // Features available on the physical device
    VkPhysicalDeviceFeatures deviceFeatures{};
    /** @brief Set of physical device features to be enabled for this example (must be set in the derived constructor) */
    VkPhysicalDeviceFeatures enabledFeatures{};
    // Features all available memory types for the physical device
    VkPhysicalDeviceMemoryProperties deviceMemoryProperties{};
    /** @brief Set of device extensions to be enabled for this example (must be set in the derived constructor) */

    std::vector<const char *> enabledDeviceExtensions;
    std::vector<const char *> enabledInstanceExtensions;
    /** @brief Logical device, application's view of the physical device (GPU) */
    VkDevice device{};
    // Handle to the device graphics queue that command buffers are submitted to
    VkQueue queue{};
    // Depth buffer format (selected during Vulkan initialization)
    VkFormat depthFormat;
    // Wraps the swap chain to present images (framebuffers) to the windowing system
    VulkanSwapchain swapchain;
    // Synchronization semaphores
    struct {
        // Swap chain image presentation
        VkSemaphore presentComplete;
        // Command buffer submission and execution
        VkSemaphore renderComplete;
    } semaphores{};
    std::vector<VkFence> waitFences;
    /** @brief Pipeline stages used to wait at for graphics queue submissions */
    VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // Contains command buffers and semaphores to be presented to the queue
    VkSubmitInfo submitInfo{};
    // CommandPool for render command buffers
    VkCommandPool cmdPool{};
    // Command buffers used for rendering
    std::vector<VkCommandBuffer> drawCmdBuffers;
    // Global render pass for frame buffer writes
    VkRenderPass renderPass;
    // List of available frame buffers (same as number of swap chain images)
    std::vector<VkFramebuffer>frameBuffers;
    // Active frame buffer index
    uint32_t currentBuffer = 0;
    // Descriptor set pool
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    // Pipeline cache object
    VkPipelineCache pipelineCache;

    // Handle to Debug Utils
    VkDebugUtilsMessengerEXT debugUtilsMessenger{};

    int frameCounter = 0;
private:
    bool viewUpdated = false;
    uint32_t destWidth;
    uint32_t destHeight;
    uint32_t lastFPS;

    void windowResize();
    void updateOverlay();

    static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void resizeCallback(GLFWwindow* window, int width, int height);
    static void cursorPositionCallback(GLFWwindow *window, double xPos, double yPos);
    static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    static void charCallback(GLFWwindow *window, unsigned int codepoint);

    void createCommandPool();
    void createCommandBuffers();
    void createSynchronizationPrimitives();
    void createPipelineCache();
    void destroyCommandBuffers();

    void setWindowSize(uint32_t width, uint32_t height);
    /** @brief Default function to handle cursor position input, calls the override function mouseMoved(...) **/
    void handleMouseMove(int32_t x, int32_t y);

    static VkPhysicalDevice pickPhysicalDevice(std::vector<VkPhysicalDevice> devices);

    static int ImGui_ImplGlfw_TranslateUntranslatedKey(int key, int scancode);

    static ImGuiKey ImGui_ImplGlfw_KeyToImGuiKey(int key);


};


#endif //MULTISENSE_VULKANRENDERER_H
