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

#include "Viewer/Core/VulkanRenderer.h"
#include "Viewer/Scripts/Private/Base.h"
#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/CRLCamera/CameraConnection.h"
#include "Generated/ScriptHeader.h"

class Renderer : VkRender::VulkanRenderer {

public:

    /**
     * @brief Default constructor for renderer
     * @param title Title of application
     */
    explicit Renderer(const std::string &title) : VulkanRenderer(title, true) {
        this->m_Title = title;
        // Create Log C++ Interface
        pLogger = Log::Logger::getInstance();
        Log::LOG_ALWAYS("<=============================== START OF PROGRAM ===============================>");

        VulkanRenderer::initVulkan();
        VulkanRenderer::prepare();
        backendInitialized = true;
        pLogger->info("Initialized Backend");

        guiManager = std::make_unique<VkRender::GuiManager>(vulkanDevice.get(), renderPass, m_Width, m_Height);

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

    std::unique_ptr<VkRender::MultiSense::CameraConnection> cameraConnection{};
    VkRender::RenderData renderData{};
    bool renderSelectionPass = true;

    // Create a host-visible staging buffer that contains the raw m_Image data
    VkBuffer selectionBuffer{};
    VkDeviceMemory selectionMemory{};
    VkBufferImageCopy bufferCopyRegion{};
    VkMemoryRequirements m_MemReqs{};

    glm::vec3 defaultCameraPosition = glm::vec3(-0.4f, 0.25f, -0.55f);
    float yaw = -270.0f, pitch = 0.0f;
    glm::vec3 defaultCameraRotation = glm::vec3(0.0f, 0.0f , 0.0f);

    void windowResized() override;
    void addDeviceFeatures() override;
    void buildCommandBuffers() override;
    void mouseMoved(float x, float y, bool&handled) override;
    void mouseScroll(float change) override;
    /**
     * @brief creates instances from classes located in src/Scripts/Objects directory.
     * Usually each class here represents object(s) in the scene
     */
    void buildScript(const std::string &scriptName);

    /**
     * @brief deletes a script if stored in \refitem builtScriptNames
     * @param scriptName m_Name of script to delete
     */
    void deleteScript(const std::string &scriptName);

    void createSelectionFramebuffer();
    void createSelectionImages();
    void destroySelectionBuffer();
    void createSelectionBuffer();
};


#endif //AR_ENGINE_RENDERER_H
