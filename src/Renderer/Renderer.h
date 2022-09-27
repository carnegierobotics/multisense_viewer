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

#include <MultiSense/src/Core/VulkanRenderer.h>
#include <MultiSense/src/Scripts/Base.h>
#include <MultiSense/src/Scripts/ScriptBuilder.h>
#include <MultiSense/src/Tools/Macros.h>
#include <MultiSense/src/CRLCamera/CameraConnection.h>

// Include Scripts
//
//#include <MultiSense/src/Scripts/Example.h>

#include <MultiSense/src/Scripts/objects/MultiSenseCamera.h>
#include <MultiSense/src/Scripts/pointcloud/PointCloud.h>
#include <MultiSense/src/Scripts/video/physical/Single/SingleLayout.h>
#include <MultiSense/src/Scripts/video/physical/Double/DoubleLayout.h>
#include <MultiSense/src/Scripts/video/physical/Double/DoubleLayoutBot.h>
#include <MultiSense/src/Scripts/video/physical/Quad/Three.h>
#include <MultiSense/src/Scripts/video/physical/Quad/PreviewOne.h>
#include <MultiSense/src/Scripts/video/physical/Quad/PreviewTwo.h>
#include <MultiSense/src/Scripts/video/physical/Quad/Four.h>

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

        prepareRenderer();
        pLogger->info("Prepared Renderer");

    };

    ~Renderer() override{
        delete guiManager;
    }

    void render() override;

    void prepareRenderer();

    void cleanUp();

    void run() {
        VulkanRenderer::renderLoop();
        cleanUp();
        destroySelectionBuffer();
    }

private:

    std::map<std::string, std::unique_ptr<Base>> scripts;
    std::unique_ptr<CameraConnection> cameraConnection{};
    std::vector<std::string> scriptNames;
    Base::Render renderData{};
    bool renderSelectionPass = false;
    // Create a host-visible staging buffer that contains the raw image data
    VkBuffer selectionBuffer{};
    VkDeviceMemory selectionMemory{};
    VkBufferImageCopy bufferCopyRegion{};
    VkMemoryRequirements memReqs{};
protected:

    glm::vec3 defaultCameraPosition = glm::vec3(0.025f, 0.15f, -0.5f);
    glm::vec3 defaultCameraRotation = glm::vec3(-4.4f, -3.2f , 0.0f);

    void windowResized() override;

    void addDeviceFeatures() override;

    void buildCommandBuffers() override;

    /**
     * Overrides UIUpdate function. Is called with an per-frame updated handle to \ref UISettings
     * @param uiSettings Handle to UISetting variables
     */
    void UIUpdate(AR::GuiObjectHandles *uiSettings) override;

    /**
     *
     * \brief creates instances from classes located in src/Scripts directory. Usually each class here represents object(s) in the scene
     */
    void deleteScript(const std::string &scriptName);

    void buildScript(const std::string &scriptName);

    void createSelectionFramebuffer();

    void createSelectionImages();

    void destroySelectionBuffer();

    void createSelectionBuffer();
};


#endif //AR_ENGINE_RENDERER_H
