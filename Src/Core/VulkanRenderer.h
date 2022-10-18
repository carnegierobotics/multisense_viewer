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

#include <MultiSense/Src/imgui/GuiManager.h>
#include <MultiSense/Src/Tools/Logger.h>

#include "VulkanSwapchain.h"
#include "Validation.h"
#include "VulkanDevice.h"


class VulkanRenderer {

public:
    explicit VulkanRenderer(const std::string &title, bool enableValidation = false);

    ~VulkanRenderer();

    /** @brief Setup the vulkan instance, flashing required extensions and connect to the physical device (GPU) */
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
    /** @brief This application is written against Vulkan API v.1.2+ **/
    uint32_t apiVersion = VK_API_VERSION_1_2;
    bool backendInitialized = false;
    uint32_t width = 1920;      // Default values - Actual values set in constructor
    uint32_t height = 1080;     // Default values - Actual values set in constructor

    /** @brief Encapsulated physical and logical vulkan device */
    std::unique_ptr<VulkanDevice> vulkanDevice{};

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

    VkRender::MouseButtons mouseButtons{};
    float mouseScrollSpeed = 50.0f;

    int keyPress = -1;
    int keyAction = -1;
    Input input;

    /** @brief Handle for Logging*/
    Log::Logger* pLogger = nullptr; // Create the object pointer for Logger Class


    /** @brief Handle for UI updates and overlay */

    /** @brief (Virtual) Creates the application wide Vulkan instance */
    virtual VkResult createInstance(bool enableValidation);
    /** @brief (Pure virtual) Render function to be implemented by the application */
    virtual void render() = 0;
    /** @brief (Virtual) Called when the camera view has changed */
    virtual void viewChanged();
    /** @brief (Virtual) Called after the mouse cursor moved and before internal events (like camera rotation) is firstUpdate */
    virtual void mouseMoved(double x, double y, bool &handled);
    /** @brief (Virtual) Called when the window has been resized, can be used by the sample application to recreate resources */
    virtual void windowResized();
    /** @brief (Virtual) Called when resources have been recreated that require a rebuild of the command buffers (e.g. frame buffer), to be implemented by the sample application */
    virtual void buildCommandBuffers();
    /** @brief (Virtual) Setup default depth and stencil views */
    virtual void setupDepthStencil();
    /** @brief (Virtual) Setup default framebuffers for all requested swapchain images */
    virtual void setupMainFramebuffer();
    /** @brief (Virtual) Setup a default renderpass */
    virtual void setupRenderPass();
    /** @brief (Virtual) Called after the physical device features have been read, can be used to set features to flashing on the device */
    virtual void addDeviceFeatures() = 0;
    /** @brief Prepares all Vulkan resources and functions required to run the sample */
    virtual void prepare();
    /** @brief Entry point for the main render loop */
    void renderLoop();
    /** Prepare the next frame for workload submission by acquiring the next swap chain image */
    void prepareFrame();
    /** @brief Presents the current image to the swap chain */
    void submitFrame();

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
    VkFormat depthFormat{};
    // Wraps the swap chain to present images (framebuffers) to the windowing system
    std::unique_ptr<VulkanSwapchain> swapchain;
    // Synchronization semaphores
    struct {
        // Swap chain image presentation
        VkSemaphore presentComplete;
        // Command buffer submission and execution
        VkSemaphore renderComplete;
    } semaphores{};
    std::vector<VkFence> waitFences{};
    /** @brief Pipeline stages used to wait at for graphics queue submissions */
    VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // Contains command buffers and semaphores to be presented to the queue
    VkSubmitInfo submitInfo{};
    // CommandPool for render command buffers
    VkCommandPool cmdPool{};
    // Command buffers used for rendering
    std::vector<VkCommandBuffer> drawCmdBuffers{};
    // Global render pass for frame buffer writes
    VkRenderPass renderPass{};
    // List of available frame buffers (same as number of swap chain images)
    std::vector<VkFramebuffer>frameBuffers{};
    // Active frame buffer index
    uint32_t currentBuffer = 0;
    // Descriptor set pool
    // Pipeline cache object
    VkPipelineCache pipelineCache{};
    VkRender::ObjectPicking selection{};
    // Handle to Debug Utils
    VkDebugUtilsMessengerEXT debugUtilsMessenger{};

    int frameCounter = 0;
    int frameID = 0;

private:
    uint32_t destWidth{};
    uint32_t destHeight{};
    float lastFPS{};

    void windowResize();
    static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void resizeCallback(GLFWwindow* window, int width, int height);
    static void cursorPositionCallback(GLFWwindow *window, double xPos, double yPos);
    static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    static void charCallback(GLFWwindow *window, unsigned int codepoint);
    static void mouseScrollCallback(GLFWwindow *window, double xoffset, double yoffset);

    void createCommandPool();
    void createCommandBuffers();
    void createSynchronizationPrimitives();
    void createPipelineCache();
    void destroyCommandBuffers();

    void setWindowSize(uint32_t width, uint32_t height);
    /** @brief Default function to handle cursor position input, calls the override function mouseMoved(...) **/
    void handleMouseMove(int32_t x, int32_t y);

    [[nodiscard]] VkPhysicalDevice pickPhysicalDevice(std::vector<VkPhysicalDevice> devices) const;

    static int ImGui_ImplGlfw_TranslateUntranslatedKey(int key, int scancode);

    static ImGuiKey ImGui_ImplGlfw_KeyToImGuiKey(int key);


};


#endif //MULTISENSE_VULKANRENDERER_H
