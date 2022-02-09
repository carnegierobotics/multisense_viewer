//
// Created by magnus on 9/4/21.
//

#ifndef AR_ENGINE_RENDERER_H
#define AR_ENGINE_RENDERER_H


// System
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <thread>
#include <fstream>
#include <filesystem>



#include <MultiSense/src/core/VulkanRenderer.h>
#include "MultiSense/src/imgui/UISettings.h"

#include "shaderParams.h"

class Renderer : VulkanRenderer {

public:


    explicit Renderer(const std::string &title) : VulkanRenderer(title, true) {
        // During constructor prepare backend for rendering
        this->title = title;
        VulkanRenderer::initVulkan();
        VulkanRenderer::prepare();
        backendInitialized = true;

        prepareRenderer();
    };
     ~Renderer() override = default;

    void render() override;
    void prepareRenderer();
    void run() {
        renderLoop();
    }

    void draw();

private:


protected:

    UBOMatrix *UBOVert{};
    FragShaderParams *UBOFrag{};


    void updateUniformBuffers();
    void prepareUniformBuffers();

    void viewChanged() override;
    void addDeviceFeatures() override;
    void buildCommandBuffers() override;
    void UIUpdate(UISettings uiSettings) override;

    void generateScriptClasses();

    void storeDepthFrame();
};


#endif //AR_ENGINE_RENDERER_H
