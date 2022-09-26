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

#include <MultiSense/src/core/VulkanRenderer.h>
#include <MultiSense/src/core/Base.h>
#include <MultiSense/src/core/ScriptBuilder.h>
#include <MultiSense/src/tools/Macros.h>
#include <MultiSense/src/core/CameraConnection.h>

// Include scripts
//
//#include <MultiSense/src/scripts/Example.h>

#include <MultiSense/src/scripts/objects/MultiSenseCamera.h>
#include <MultiSense/src/scripts/pointcloud/PointCloud.h>
#include <MultiSense/src/scripts/video/physical/Single/SingleLayout.h>
#include <MultiSense/src/scripts/video/physical/Double/DoubleLayout.h>
#include <MultiSense/src/scripts/video/physical/Double/DoubleLayoutBot.h>
#include <MultiSense/src/scripts/video/physical/Quad/Three.h>
#include <MultiSense/src/scripts/video/physical/Quad/PreviewOne.h>
#include <MultiSense/src/scripts/video/physical/Quad/PreviewTwo.h>
#include <MultiSense/src/scripts/video/physical/Quad/Four.h>

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
        renderLoop();
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
    VkBuffer selectionBuffer;
    VkDeviceMemory selectionMemory;
    VkBufferImageCopy bufferCopyRegion{};
    VkMemoryRequirements memReqs{};
protected:

    glm::vec3 defaultCameraPosition = glm::vec3(2.0f, 1.2f, -5.0f);
    glm::vec3 defaultCameraRotation = glm::vec3(0.0f, 0.0f, 0.0f);

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
     * \brief creates instances from classes located in src/scripts directory. Usually each class here represents object(s) in the scene
     */
    void generateScriptClasses();

    void deleteScript(const std::string &scriptName);

    void buildScript(const std::string &scriptName);

    void createSelectionFramebuffer();

    void createSelectionImages();

    void destroySelectionBuffer();

    void createSelectionBuffer();
};


#endif //AR_ENGINE_RENDERER_H
