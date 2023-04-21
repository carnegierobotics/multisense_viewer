/**
 * @file: MultiSense-Viewer/include/Viewer/Renderer/Renderer.h
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
 *   2021-9-4, mgjerde@carnegierobotics.com, Created file.
 **/
#ifndef MULTISENSE_RENDERER_H
#define MULTISENSE_RENDERER_H

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
#include "Viewer/Core/Skybox.h"
#include "Viewer/Renderer/UsageMonitor.h"
#include "Viewer/Core/RendererConfig.h"

class Renderer : VkRender::VulkanRenderer {

public:

    /**
     * @brief Default constructor for renderer
     * @param title Title of application
     */
    explicit Renderer(const std::string &title) : VulkanRenderer(title, true) {
        VkRender::RendererConfig& config = VkRender::RendererConfig::getInstance();
        this->m_Title = title;
        // Create Log C++ Interface
        pLogger = Log::Logger::getInstance();
        // Start up usage monitor
        usageMonitor = std::make_unique<UsageMonitor>();

        VulkanRenderer::initVulkan();
        VulkanRenderer::prepare();
        backendInitialized = true;
        pLogger->info("Initialized Backend");
        config.setGpuDevice(physicalDevice);

        guiManager = std::make_unique<VkRender::GuiManager>(vulkanDevice.get(), renderPass, m_Width, m_Height);
        guiManager->handles.mouse = &mouseButtons;
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
    std::unique_ptr<UsageMonitor> usageMonitor;
    std::unique_ptr<VkRender::MultiSense::CameraConnection> cameraConnection{};
    VkRender::RenderData renderData{};
    bool renderSelectionPass = true;

    // Create a host-visible staging buffer that contains the raw m_Image data
    VkBuffer selectionBuffer{};
    VkDeviceMemory selectionMemory{};
    VkBufferImageCopy bufferCopyRegion{};
    VkMemoryRequirements m_MemReqs{};

    glm::vec3 defaultCameraPosition = glm::vec3(0.0f, 0.0f, 1.0f);
    float yaw = -270.0f, pitch = 0.0f;

    void windowResized() override;
    void addDeviceFeatures() override;
    void buildCommandBuffers() override;
    void mouseMoved(float x, float y, bool&handled) override;
    void mouseScroll(float change) override;
    /**
     * @brief creates instances from classes located in src/Scripts/Objects directory.
     * Usually each class here represents object(s) in the scene
     */
    void buildScripts();

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


#endif // MultiSense_Viewer_RENDERER_H
