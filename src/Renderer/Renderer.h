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
#include <MultiSense/src/imgui/SideBar.h>
#include <MultiSense/src/imgui/InteractionMenu.h>

// Include scripts
//
//#include <MultiSense/src/scripts/Example.h>

#include <MultiSense/src/scripts/objects/LightSource.h>
#include <MultiSense/src/scripts/pointcloud/PointCloud.h>
#include <MultiSense/src/scripts/pointcloud/VirtualPointCloud.h>
#include <MultiSense/src/scripts/video/virtual/LeftImager.h>
#include <MultiSense/src/scripts/video/virtual/RightImager.h>
#include <MultiSense/src/scripts/video/virtual/AuxImager.h>
#include <MultiSense/src/scripts/video/physical/DefaultPreview.h>
#include <MultiSense/src/scripts/video/physical/DisparityPreview.h>
#include <MultiSense/src/scripts/video/physical/RightPreview.h>
#include <MultiSense/src/scripts/video/physical/AuxiliaryPreview.h>

class Renderer : VulkanRenderer {

public:

    /**
     * @brief Default constructor for renderer
     * @param title Title of application
     */
    explicit Renderer(const std::string &title) : VulkanRenderer(title, true) {
        this->title = title;
        // Create Log C++ Interface
        Log::LOG_ALWAYS("<=============================== START OF PROGRAM ===============================>");
        pLogger = Log::Logger::getInstance();

        VulkanRenderer::initVulkan();
        VulkanRenderer::prepare();
        backendInitialized = true;
        pLogger->info("Initialized Backend");

        prepareRenderer();
        pLogger->info("Prepared Renderer");

    };

    ~Renderer() override = default;

    void render() override;

    void prepareRenderer();

    void cleanUp();

    void run() {
        renderLoop();
        cleanUp();
    }

private:

    std::map<std::string, std::unique_ptr<Base>> scripts;
    std::unique_ptr<CameraConnection> cameraConnection{};
    std::vector<std::string> scriptNames;

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

    void buildScript(const std::string& scriptName);
};


#endif //AR_ENGINE_RENDERER_H
