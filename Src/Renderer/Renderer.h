//
// Created by magnus on 9/4/21.
//

#ifndef MULTISENSE_RENDERER_H
#define MULTISENSE_RENDERER_H


// System
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <thread>
#include <fstream>
#include <filesystem>

#include <MultiSense/Src/Core/VulkanRenderer.h>
#include <MultiSense/Src/Scripts/Private/Base.h>
#include <MultiSense/Src/Scripts/Private/ScriptBuilder.h>
#include <MultiSense/Src/Tools/Macros.h>
#include <MultiSense/Src/CRLCamera/CameraConnection.h>

#include "MultiSense/Assets/Generated/ScriptHeader.h"

class Renderer : VulkanRenderer {

public:

    /**
     * @brief Default constructor for renderer
     * @param title Title of application
     */
    explicit Renderer(const std::string &title) : VulkanRenderer(title, true) {
        this->title = title;
        // Create Log C++ Interface
        pLogger = Log::Logger::getInstance();
        Log::LOG_ALWAYS("<=============================== START OF PROGRAM ===============================>");

        VulkanRenderer::initVulkan();
        VulkanRenderer::prepare();
        backendInitialized = true;
        pLogger->info("Initialized Backend");

        guiManager = std::make_unique<VkRender::GuiManager>(vulkanDevice.get(), renderPass, width, height);

        prepareRenderer();
        pLogger->info("Prepared Renderer");

    };

    ~Renderer()= default;

    /**
     * @brief runs the renderer loop
     */
    void run() {
        VulkanRenderer::renderLoop();
    }

    /**
     * @brief cleans up resources on application exist
     */
    void cleanUp();

private:

    void render() override;

    void prepareRenderer();


    std::unique_ptr<VkRender::GuiManager> guiManager{};
    std::map<std::string, std::unique_ptr<VkRender::Base>> scripts{};
    std::vector<std::string> builtScriptNames; 

    std::unique_ptr<CameraConnection> cameraConnection{};
    VkRender::RenderData renderData{};
    bool renderSelectionPass = false;

    // Create a host-visible staging buffer that contains the raw image data
    VkBuffer selectionBuffer{};
    VkDeviceMemory selectionMemory{};
    VkBufferImageCopy bufferCopyRegion{};
    VkMemoryRequirements memReqs{};

    glm::vec3 defaultCameraPosition = glm::vec3(0.025f, 0.15f, -0.5f);
    glm::vec3 defaultCameraRotation = glm::vec3(-4.4f, -3.2f , 0.0f);

    void windowResized() override;
    void addDeviceFeatures() override;
    void buildCommandBuffers() override;
    void mouseMoved(double x, double y, bool&handled) override;

    /**
     * @brief creates instances from classes located in src/Scripts/Objects directory.
     * Usually each class here represents object(s) in the scene
     */
    void buildScript(const std::string &scriptName);

    /**
     * @brief deletes a script if stored in \refitem builtScriptNames
     * @param scriptName name of script to delete
     */
    void deleteScript(const std::string &scriptName);

    void createSelectionFramebuffer();
    void createSelectionImages();
    void destroySelectionBuffer();
    void createSelectionBuffer();
};


#endif //AR_ENGINE_RENDERER_H
