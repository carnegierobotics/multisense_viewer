/**
 * @file: MultiSense-Viewer/src/Renderer/Renderer.cpp
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
 *   2022-4-9, mgjerde@carnegierobotics.com, Created file.
 **/


#ifdef WIN32
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <sstream>
#endif

#include <array>

#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Populate.h"


Renderer::Renderer(const std::string& title) : VulkanRenderer(title) {
    VkRender::RendererConfig& config = VkRender::RendererConfig::getInstance();
    this->m_Title = title;
    // Create Log C++ Interface
    Log::Logger::getInstance()->setLogLevel(config.getLogLevel());
    pLogger = Log::Logger::getInstance();


    VulkanRenderer::initVulkan();
    VulkanRenderer::prepare();
    backendInitialized = true;
    pLogger->info("Initialized Backend");
    config.setGpuDevice(physicalDevice);

    // Start up usage monitor
    usageMonitor = std::make_shared<UsageMonitor>();
    usageMonitor->loadSettingsFromFile();
    usageMonitor->userStartSession(rendererStartTime);

    guiManager = std::make_unique<VkRender::GuiManager>(vulkanDevice.get(), renderPass, m_Width, m_Height, msaaSamples,
                                                        swapchain->imageCount);
    guiManager->handles.mouse = &mouseButtons;
    guiManager->handles.usageMonitor = usageMonitor;
    guiManager->handles.camera.type = camera.type;

    prepareRenderer();
    pLogger->info("Prepared Renderer");
}


void Renderer::prepareRenderer() {
    camera.type = VkRender::Camera::flycam;
    camera.setPerspective(60.0f, static_cast<float>(m_Width) / static_cast<float>(m_Height), 0.01f, 100.0f);
    camera.resetPosition();
    camera.resetRotation();
    createSelectionImages();
    createSelectionFramebuffer();
    createSelectionBuffer();
    cameraConnection = std::make_unique<VkRender::MultiSense::CameraConnection>();

    std::ifstream infile(Utils::getAssetsPath().append("Generated/Scripts.txt").string());
    std::string line;
    while (std::getline(infile, line)) {
        // Skip comment # line
        if (line.find('#') != std::string::npos || line.find("Skybox") != std::string::npos)
            continue;
        availableScriptNames.emplace_back(line);
    }
    // Load Object Scripts from file
    buildScript("Skybox");
    for (const auto& name : availableScriptNames)
        buildScript(name);
}

void Renderer::addDeviceFeatures() {
    if (deviceFeatures.fillModeNonSolid) {
        enabledFeatures.fillModeNonSolid = VK_TRUE;
        // Wide lines must be present for line m_Width > 1.0f
        if (deviceFeatures.wideLines) {
            enabledFeatures.wideLines = VK_TRUE;
        }
        if (deviceFeatures.samplerAnisotropy) {
            enabledFeatures.samplerAnisotropy = VK_TRUE;
        }
    }
}

void Renderer::buildScript(const std::string& scriptName) {
    bool exists = Utils::isInVector(builtScriptNames, scriptName);
    bool isInDeletionQueue = scriptsForDeletion.contains(scriptName);
    if (exists || isInDeletionQueue) {
        Log::Logger::getInstance()->warning("Script {} Already exists or is in deletion, not pushing to render graphicsQueue",
                                            scriptName);
        return;
    }

    scripts[scriptName] = VkRender::ComponentMethodFactory::Create(scriptName);
    if (scripts[scriptName].get() == nullptr) {
        pLogger->error("Failed to register script {}.", scriptName);
        builtScriptNames.erase(std::find(builtScriptNames.begin(), builtScriptNames.end(), scriptName));
        return;
    }
    pLogger->info("Registered script: {} in factory", scriptName.c_str());
    builtScriptNames.emplace_back(scriptName);

    // Run Once
    renderUtils.device = vulkanDevice.get();
    renderUtils.instance = &instance;
    renderUtils.renderPass = &renderPass;
    renderUtils.msaaSamples = msaaSamples;
    renderUtils.UBCount = swapchain->imageCount;
    renderUtils.picking = &selection;
    renderUtils.queueSubmitMutex = &queueSubmitMutex;
    renderUtils.vkDeviceUUID = vkDeviceUUID;
    renderUtils.fence = &waitFences;
    renderData.height = m_Height;
    renderData.width = m_Width;
    renderData.camera = &camera;

    // Copy data generated from TOP OF PIPE scripts
    renderUtils.skybox.irradianceCube = scripts["Skybox"]->skyboxTextures.irradianceCube;
    renderUtils.skybox.lutBrdf = scripts["Skybox"]->skyboxTextures.lutBrdf;
    renderUtils.skybox.prefilterEnv = scripts["Skybox"]->skyboxTextures.prefilterEnv;
    renderUtils.skybox.prefilteredCubeMipLevels = scripts["Skybox"]->skyboxTextures.prefilteredCubeMipLevels;

    scripts[scriptName]->createUniformBuffers(renderUtils, renderData, &topLevelScriptData);

    auto conf = VkRender::RendererConfig::getInstance().getUserSetting();
    conf.scripts.names = builtScriptNames;
    VkRender::RendererConfig::getInstance().setUserSetting(conf);
}

void Renderer::deleteScript(const std::string& scriptName) {
    if (scriptsForDeletion.contains(scriptName)) {
        Log::Logger::getInstance()->warning("Script name {} already requested deletion. Skipped deletion of script.",
                                            scriptName);
        return;
    }

    if (builtScriptNames.empty())
        return;
    auto it = std::find(builtScriptNames.begin(), builtScriptNames.end(), scriptName);
    if (it != builtScriptNames.end())
        builtScriptNames.erase(it);
    else
        return;

    pLogger->info("Pushing script to Delete queue: {}. Erased script from rendered list", scriptName.c_str());
    scriptsForDeletion[scriptName] = scripts[scriptName];
    scriptsForDeletion[scriptName]->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
    scripts.erase(scriptName);

    auto conf = VkRender::RendererConfig::getInstance().getUserSetting();
    conf.scripts.names = builtScriptNames;
    VkRender::RendererConfig::getInstance().setUserSetting(conf);
}


void Renderer::buildCommandBuffers() {
    VkCommandBufferBeginInfo cmdBufInfo = Populate::commandBufferBeginInfo();
    cmdBufInfo.flags = 0;
    cmdBufInfo.pInheritanceInfo = nullptr;

    std::array<VkClearValue, 3> clearValues{};

    if (guiManager->handles.renderer3D) {
        clearValues[0] = {
            {
                {
                    0.0f, 0.0f,
                    0.0f, 1.0f
                }
            }
        };
        clearValues[2] = {
            {
                {
                    0.0f, 0.0f,
                    0.0f, 1.0f
                }
            }
        };
    }
    else {
        clearValues[0] = {
            {
                {
                    guiManager->handles.clearColor[0], guiManager->handles.clearColor[1],
                    guiManager->handles.clearColor[2], guiManager->handles.clearColor[3]
                }
            }
        };
        clearValues[2] = {
            {
                {
                    guiManager->handles.clearColor[0], guiManager->handles.clearColor[1],
                    guiManager->handles.clearColor[2], guiManager->handles.clearColor[3]
                }
            }
        };
    }
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = m_Width;
    renderPassBeginInfo.renderArea.extent.height = m_Height;
    renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassBeginInfo.pClearValues = clearValues.data();

    const VkViewport viewport = Populate::viewport(static_cast<float>(m_Width), static_cast<float>(m_Height), 0.0f,
                                                   1.0f);
    const VkRect2D scissor = Populate::rect2D(static_cast<int32_t>(m_Width), static_cast<int32_t>(m_Height), 0, 0);

    renderPassBeginInfo.framebuffer = frameBuffers[imageIndex];
    vkBeginCommandBuffer(drawCmdBuffers.buffers[currentFrame], &cmdBufInfo);

    // Syncrhonization before renderpass

    // Image memory barrier to make sure that compute shader writes are finished before sampling from the texture
    if (topLevelScriptData.compute.valid) {
        VkImageMemoryBarrier imageMemoryBarrier = {};
        imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        // We won't be changing the layout of the image
        imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageMemoryBarrier.image = (*topLevelScriptData.compute.textureComputeTarget)[currentFrame].m_Image;
        imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(
            drawCmdBuffers.buffers[currentFrame],
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &imageMemoryBarrier);
    }

    vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdSetViewport(drawCmdBuffers.buffers[currentFrame], 0, 1, &viewport);
    vkCmdSetScissor(drawCmdBuffers.buffers[currentFrame], 0, 1, &scissor);

    // Draw scripts that must be drawn first
    for (auto& script : scripts) {
        if (script.second->getType() == VkRender::CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE)
            script.second->drawScript(&drawCmdBuffers, currentFrame, true);
    }

    /** Generate Script draw commands **/
    for (auto& script : scripts) {
        if (script.second->getType() != VkRender::CRL_SCRIPT_TYPE_DISABLED &&
            script.second->getType() != VkRender::CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE &&
            script.second->getType() != VkRender::CRL_SCRIPT_TYPE_SIMULATED_CAMERA) {
            script.second->drawScript(&drawCmdBuffers, currentFrame, true);
        }
    }
    /** Generate UI draw commands **/
    guiManager->drawFrame(drawCmdBuffers.buffers[currentFrame], currentFrame);
    vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);
    CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers.buffers[currentFrame]));
}

bool Renderer::compute() {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(computeCommand.buffers[currentFrame], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording compute command buffer!");
    }
    for (auto& script : scripts) {
        if (script.second->getType() & VkRender::CRL_SCRIPT_TYPE_SIMULATED_CAMERA) {
            script.second->drawScript(&computeCommand, currentFrame, true);
        }
    }
    if (vkEndCommandBuffer(computeCommand.buffers[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record compute command buffer!");
    }
    return true;
}

void Renderer::updateUniformBuffers() {
    renderData.camera = &camera;
    renderData.deltaT = frameTimer;
    renderData.index = currentFrame;
    renderData.height = m_Height;
    renderData.width = m_Width;
    renderData.crlCamera = &cameraConnection->camPtr;

    // Delete the requested scripts if resources are no longer busy in render pipeline
    std::vector<std::string> scriptsToDelete;
    // Reload scripts if requested
    std::vector<std::string> scriptsToReload;

    for (const auto& script : scriptsForDeletion) {
        scriptsToDelete.push_back(script.first);
    }
    for (const auto& scriptName : scriptsToDelete) {
        if (scriptsForDeletion[scriptName]->onDestroyScript()) {
            scriptsForDeletion[scriptName].reset();
            scriptsForDeletion.erase(scriptName);
            scriptsToReload.emplace_back(scriptName);
        }
    }

    for (const auto& script : scripts) {
        if (script.second->getDrawMethod() == VkRender::CRL_SCRIPT_RELOAD) {
            scriptsToReload.push_back(script.first);
        }
    }

    for (const auto& scriptName : scriptsToReload) {
        if (scriptName == "Skybox") {
            // Clear script and scriptnames before rebuilding
            for (const auto& name : builtScriptNames) {
                pLogger->info("Deleting Script: {}", name.c_str());
                scripts[name].get()->onDestroyScript();
                scripts[name].reset();
                scripts.erase(name);
            }
            builtScriptNames.clear();
            buildScript("Skybox");
            for (const auto& name : availableScriptNames)
                buildScript(name);
        }
        else {
            deleteScript(scriptName);
            buildScript(scriptName);
        }
    }
    // New version available?
    std::string versionRemote;
    if (guiManager->handles.askUserForNewVersion && usageMonitor->getLatestAppVersionRemote(&versionRemote)) {
        std::string localAppVersion = VkRender::RendererConfig::getInstance().getAppVersion();
        Log::Logger::getInstance()->info("New Version is Available: Local version={}, available version={}",
                                         localAppVersion, versionRemote);
        guiManager->handles.newVersionAvailable = Utils::isLocalVersionLess(localAppVersion, versionRemote);
    }
    pLogger->frameNumber = frameID;
    if (keyPress == GLFW_KEY_SPACE) {
        camera.resetPosition();
        camera.resetRotation();
    }
    guiManager->handles.camera.pos = glm::vec3(glm::inverse(camera.matrices.view)[3]);
    guiManager->handles.camera.up = camera.cameraUp;
    guiManager->handles.camera.target = camera.m_Target;
    guiManager->handles.camera.cameraFront = camera.cameraFront;

    // Update GUI
    guiManager->handles.info->frameID = frameID;
    guiManager->update((frameCounter == 0), frameTimer, renderData.width, renderData.height, &input);

    // Update Camera connection based on Actions from GUI
    cameraConnection->onUIUpdate(guiManager->handles.devices, guiManager->handles.configureNetwork);
    // Enable/disable Renderer3D scripts and simulated camera

    for (auto& script : scripts) {
        if (!guiManager->handles.renderer3D) {
            if (script.second->getType() & VkRender::CRL_SCRIPT_TYPE_RENDERER3D)
                script.second->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
        }
        else {
            if (script.second->getType() & VkRender::CRL_SCRIPT_TYPE_RENDERER3D)
                script.second->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);
        }
        // TODO add simulated camera handling
    }


    if (guiManager->handles.camera.type == 0)
        camera.type = VkRender::Camera::arcball;
    if (guiManager->handles.camera.type == 1)
        camera.type = VkRender::Camera::flycam;
    if (guiManager->handles.camera.reset) {
        camera.resetPosition();
        camera.resetRotation();
    }

    // Enable scripts depending on gui layout chosen
    bool noActivePreview = true;
    for (auto& dev : guiManager->handles.devices) {
        if (dev.state == VkRender::CRL_STATE_ACTIVE) {
            noActivePreview = false;
            switch (dev.layout) {
            case VkRender::CRL_PREVIEW_LAYOUT_SINGLE:
                scripts.at("SingleLayout")->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);
                scripts.at("DoubleTop")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("DoubleBot")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("One")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("Two")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("Three")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("Four")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                break;
            case VkRender::CRL_PREVIEW_LAYOUT_DOUBLE:
                scripts.at("DoubleTop")->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);
                scripts.at("DoubleBot")->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);

                scripts.at("SingleLayout")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("One")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("Two")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("Three")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("Four")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                break;
            case VkRender::CRL_PREVIEW_LAYOUT_QUAD:
                scripts.at("One")->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);
                scripts.at("Two")->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);
                scripts.at("Three")->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);
                scripts.at("Four")->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);

                scripts.at("SingleLayout")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("DoubleTop")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("DoubleBot")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                break;
            default:
                scripts.at("SingleLayout")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("DoubleTop")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("DoubleBot")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("One")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("Two")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("Three")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                scripts.at("Four")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                break;
            }

            if (scripts.contains("MultiSenseCamera") && scripts.contains("PointCloud")) { // TODO quickfix to check if script exists during reload. Need a more permament fix
                switch (dev.selectedPreviewTab) {
                    case VkRender::CRL_TAB_3D_POINT_CLOUD:
                        scripts.at("PointCloud")->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);
                        scripts.at("MultiSenseCamera")->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);
                        scripts.at("Skybox")->setDrawMethod(VkRender::CRL_SCRIPT_DRAW);
                        break;
                    default:
                        scripts.at("PointCloud")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                        scripts.at("Skybox")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                        scripts.at("MultiSenseCamera")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
                        break;
                }
            }
        }
    }
    if (noActivePreview) {
        scripts.at("SingleLayout")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
        scripts.at("DoubleTop")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
        scripts.at("DoubleBot")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
        scripts.at("One")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
        scripts.at("Two")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
        scripts.at("Three")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
        scripts.at("Four")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
        scripts.at("PointCloud")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
        scripts.at("MultiSenseCamera")->setDrawMethod(VkRender::CRL_SCRIPT_DONT_DRAW);
    }
    // Run update function on active camera Scripts and build them if not built
    for (size_t i = 0; i < guiManager->handles.devices.size(); ++i) {
        if (guiManager->handles.devices.at(i).state == VkRender::CRL_STATE_REMOVE_FROM_LIST)
            guiManager->handles.devices.erase(guiManager->handles.devices.begin() + i);
    }

    // Scripts that share dataa
    for (auto& script : scripts) {
        if (script.second->getType() != VkRender::CRL_SCRIPT_TYPE_DISABLED) {
            if (!script.second->sharedData->destination.empty()) {
                // Send to destination script
                if (script.second->sharedData->destination == "All") {
                    // Copy shared data to all
                    auto& shared = script.second->sharedData;

                    for (auto& s : scripts) {
                        if (s == script)
                            continue;
                        memcpy(s.second->sharedData->data, shared->data, SHARED_MEMORY_SIZE_1MB);
                    }
                }
            }
        }
    }

    // UIUpdateFunction on Scripts with const handle to GUI
    for (auto& script : scripts) {
        if (script.second->getType() != VkRender::CRL_SCRIPT_TYPE_DISABLED)
            script.second->uiUpdate(&guiManager->handles);
    }
    // run update function for camera connection
    for (auto& dev : guiManager->handles.devices) {
        if (dev.state != VkRender::CRL_STATE_ACTIVE)
            continue;
        cameraConnection->update(dev);
    }


    // Update renderer with application settings
    auto conf = VkRender::RendererConfig::getInstance().getUserSetting();
    for (auto& script : conf.scripts.rebuildMap) {
        if (script.second) {
            // if rebuild
            scripts.at(script.first)->setDrawMethod(VkRender::CRL_SCRIPT_RELOAD);
            script.second = false;
        }
    }
    VkRender::RendererConfig::getInstance().setUserSetting(conf);


    // Run update function on Scripts
    for (auto& script : scripts) {
        if (script.second->getType() != VkRender::CRL_SCRIPT_TYPE_DISABLED) {
            script.second->updateUniformBufferData(&renderData);
        }
    }
}


void Renderer::recordCommands() {
    /** Generate Draw Commands **/
    buildCommandBuffers();
    /** IF WE SHOULD RENDER SECOND IMAGE FOR MOUSE PICKING EVENTS (Reason: let user see PerPixelInformation)
     *  THIS INCLUDES RENDERING SELECTED OBJECTS AND COPYING CONTENTS BACK TO CPU INSTEAD OF DISPLAYING TO SCREEN **/
    if (renderSelectionPass) {
        CommandBuffer cmdBuffer{};
        cmdBuffer.buffers.emplace_back(vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true));
        std::array<VkClearValue, 3> clearValues{};
        clearValues[0] = {
            {
                {
                    guiManager->handles.clearColor[0], guiManager->handles.clearColor[1],
                    guiManager->handles.clearColor[2], guiManager->handles.clearColor[3]
                }
            }
        };
        clearValues[2] = {
            {
                {
                    guiManager->handles.clearColor[0], guiManager->handles.clearColor[1],
                    guiManager->handles.clearColor[2], guiManager->handles.clearColor[3]
                }
            }
        };
        clearValues[1].depthStencil = {1.0f, 0};
        const VkViewport viewport = Populate::viewport(static_cast<float>(m_Width), static_cast<float>(m_Height), 0.0f,
                                                       1.0f);
        const VkRect2D scissor = Populate::rect2D(static_cast<int32_t>(m_Width), static_cast<int32_t>(m_Height), 0,
                                                  0);

        VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.offset.x = 0;
        renderPassBeginInfo.renderArea.offset.y = 0;
        renderPassBeginInfo.renderArea.extent.width = m_Width;
        renderPassBeginInfo.renderArea.extent.height = m_Height;
        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassBeginInfo.pClearValues = clearValues.data();
        renderPassBeginInfo.renderPass = selection.renderPass;
        renderPassBeginInfo.framebuffer = selection.frameBuffer;
        vkCmdBeginRenderPass(cmdBuffer.buffers.front(), &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmdBuffer.buffers.front(), 0, 1, &viewport);
        vkCmdSetScissor(cmdBuffer.buffers.front(), 0, 1, &scissor);
        for (auto& script : scripts) {
            if (script.second->getType() != VkRender::CRL_SCRIPT_TYPE_DISABLED) {
                script.second->drawScript(&cmdBuffer, 0, false);
            }
        }
        vkCmdEndRenderPass(cmdBuffer.buffers.front());
        vulkanDevice->flushCommandBuffer(cmdBuffer.buffers.front(), graphicsQueue);
        VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkMemoryBarrier memoryBarrier = {
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_READ_BIT, // srcAccessMask
            VK_ACCESS_HOST_READ_BIT
        }; // dstAccessMask
        vkCmdPipelineBarrier(
            copyCmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, // srcStageMask
            VK_PIPELINE_STAGE_HOST_BIT, // dstStageMask
            VK_DEPENDENCY_BY_REGION_BIT,
            1, // memoryBarrierCount
            &memoryBarrier,
            0, nullptr,
            0, nullptr); // pMemoryBarriers);

        // Copy mip levels from staging buffer
        vkCmdCopyImageToBuffer(
            copyCmd,
            selection.colorImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            selectionBuffer,
            1,
            &bufferCopyRegion
        );
        vulkanDevice->flushCommandBuffer(copyCmd, graphicsQueue);
        // Copy texture data into staging buffer
        uint8_t* data = nullptr;
        CHECK_RESULT(
            vkMapMemory(vulkanDevice->m_LogicalDevice, selectionMemory, 0, m_MemReqs.size, 0,
                reinterpret_cast<void **>(&data)));
        vkUnmapMemory(vulkanDevice->m_LogicalDevice, selectionMemory);
        for (auto& dev : guiManager->handles.devices) {
            if (dev.state != VkRender::CRL_STATE_ACTIVE)
                continue;

            if (dev.simulatedDevice) {
                for (auto& win : dev.win) {
                    // Skip second render pass if we dont have a source selected or if the source is point cloud related
                    if (!win.second.isHovered ||
                        win.first == VkRender::CRL_PREVIEW_POINT_CLOUD)
                        continue;

                    auto windowIndex = win.first;
                    float viewAreaElementPosX = win.second.xPixelStartPos;
                    float viewAreaElementPosY = win.second.yPixelStartPos;
                    float imGuiPosX = mousePos.x - viewAreaElementPosX -
                        (guiManager->handles.info->previewBorderPadding / 2.0f);
                    float imGuiPosY = mousePos.y - viewAreaElementPosY -
                        (guiManager->handles.info->previewBorderPadding / 2.0f);
                    float maxInRangeX = guiManager->handles.info->viewAreaElementSizeX -
                        guiManager->handles.info->previewBorderPadding;
                    float maxInRangeY = guiManager->handles.info->viewAreaElementSizeY -
                        guiManager->handles.info->previewBorderPadding;
                    if (imGuiPosX > 0 && imGuiPosX < maxInRangeX
                        && imGuiPosY > 0 && imGuiPosY < maxInRangeY) {
                        auto x = static_cast<uint32_t>(static_cast<float>(1920) * (imGuiPosX) / maxInRangeX);
                        auto y = static_cast<uint32_t>(static_cast<float>(1080) * (imGuiPosY) / maxInRangeY);
                        // Add one since we are not counting from zero anymore :)
                        dev.pixelInfo[windowIndex].x = x + 1;
                        dev.pixelInfo[windowIndex].y = y + 1;

                        // Check that we are within bounds
                        if (dev.pixelInfoZoomed[windowIndex].y > 1080)
                            dev.pixelInfoZoomed[windowIndex].y = 0;
                        if (dev.pixelInfoZoomed[windowIndex].x > 1920)
                            dev.pixelInfoZoomed[windowIndex].x = 0;
                    }
                }
                continue;
            }
            auto idx = static_cast<uint32_t>((mousePos.x + (m_Width * mousePos.y)) * 4);
            if (idx > m_Width * m_Height * 4)
                continue;

            if (dev.state == VkRender::CRL_STATE_ACTIVE) {
                for (auto& win : dev.win) {
                    // Skip second render pass if we dont have a source selected or if the source is point cloud related
                    if (!win.second.isHovered || win.second.selectedSource == "Idle" ||
                        win.first == VkRender::CRL_PREVIEW_POINT_CLOUD)
                        continue;

                    auto windowIndex = win.first;
                    auto tex = VkRender::TextureData(Utils::CRLSourceToTextureType(win.second.selectedSource),
                                                     dev.channelInfo[win.second.selectedRemoteHeadIndex].
                                                     selectedResolutionMode,
                                                     true);

                    if (renderData.crlCamera->getCameraStream(win.second.selectedSource, &tex,
                                                              win.second.selectedRemoteHeadIndex)) {
                        uint32_t width = 0, height = 0, depth = 0;
                        Utils::cameraResolutionToValue(
                            dev.channelInfo[win.second.selectedRemoteHeadIndex].selectedResolutionMode,
                            &width, &height, &depth);

                        float viewAreaElementPosX = win.second.xPixelStartPos;
                        float viewAreaElementPosY = win.second.yPixelStartPos;
                        float imGuiPosX = mousePos.x - viewAreaElementPosX -
                            (guiManager->handles.info->previewBorderPadding / 2.0f);
                        float imGuiPosY = mousePos.y - viewAreaElementPosY -
                            (guiManager->handles.info->previewBorderPadding / 2.0f);
                        float maxInRangeX = guiManager->handles.info->viewAreaElementSizeX -
                            guiManager->handles.info->previewBorderPadding;
                        float maxInRangeY = guiManager->handles.info->viewAreaElementSizeY -
                            guiManager->handles.info->previewBorderPadding;
                        if (imGuiPosX > 0 && imGuiPosX < maxInRangeX
                            && imGuiPosY > 0 && imGuiPosY < maxInRangeY) {
                            uint32_t w = 0, h = 0, d = 0;
                            Utils::cameraResolutionToValue(
                                dev.channelInfo[win.second.selectedRemoteHeadIndex].selectedResolutionMode, &w,
                                &h,
                                &d);

                            auto x = static_cast<uint32_t>(static_cast<float>(w) * (imGuiPosX) / maxInRangeX);
                            auto y = static_cast<uint32_t>(static_cast<float>(h) * (imGuiPosY) / maxInRangeY);
                            // Add one since we are not counting from zero anymore :)
                            dev.pixelInfo[windowIndex].x = x + 1;
                            dev.pixelInfo[windowIndex].y = y + 1;

                            // Check that we are within bounds
                            if (dev.pixelInfoZoomed[windowIndex].y > h)
                                dev.pixelInfoZoomed[windowIndex].y = 0;
                            if (dev.pixelInfoZoomed[windowIndex].x > w)
                                dev.pixelInfoZoomed[windowIndex].x = 0;

                            switch (Utils::CRLSourceToTextureType(win.second.selectedSource)) {
                            case VkRender::CRL_GRAYSCALE_IMAGE: {
                                Log::Logger::getInstance()->traceWithFrequency("Selection_grayscale_tag", 10,
                                                                               "Calculating hovered pixel intensity, res: {}x{}x{}, pos: {},{} posZoomed: {}, {}",
                                                                               w, h, d,
                                                                               dev.pixelInfo[windowIndex].x,
                                                                               dev.pixelInfo[windowIndex].y,
                                                                               dev.pixelInfoZoomed[windowIndex].x,
                                                                               dev.pixelInfoZoomed[windowIndex].y);

                                uint8_t intensity = tex.data[(w * y) + x];
                                dev.pixelInfo[windowIndex].intensity = intensity;

                                intensity = tex.data[(w * dev.pixelInfoZoomed[windowIndex].y) +
                                    dev.pixelInfoZoomed[windowIndex].x];
                                dev.pixelInfoZoomed[windowIndex].intensity = intensity;
                            }
                            break;
                            case VkRender::CRL_DISPARITY_IMAGE: {
                                float disparity = 0;
                                auto* p = reinterpret_cast<uint16_t*>(tex.data);
                                disparity = p[(w * y) + x] / 16.0f;
                                Log::Logger::getInstance()->traceWithFrequency("Selection_disparity_tag", 10,
                                                                               "Calculating hovered pixel distance, res: {}x{}x{}, pos: {},{} posZoomed: {}, {}",
                                                                               w, h, d,
                                                                               dev.pixelInfo[windowIndex].x,
                                                                               dev.pixelInfo[windowIndex].y,
                                                                               dev.pixelInfoZoomed[windowIndex].x,
                                                                               dev.pixelInfoZoomed[windowIndex].y);
                                // get focal length
                                float fx = cameraConnection->camPtr.getCameraInfo(
                                    win.second.selectedRemoteHeadIndex).calibration.left.P[0][0];
                                float tx = cameraConnection->camPtr.getCameraInfo(
                                        win.second.selectedRemoteHeadIndex).calibration.right.P[0][3] /
                                    (fx * (1920.0f / static_cast<float>(w)));
                                if (disparity > 0) {
                                    float dist = (fx * abs(tx)) / disparity;
                                    dev.pixelInfo[windowIndex].depth = dist;
                                }
                                else {
                                    dev.pixelInfo[windowIndex].depth = 0;
                                }
                                auto disparityDisplayed =
                                    p[(w * dev.pixelInfoZoomed[windowIndex].y) +
                                        dev.pixelInfoZoomed[windowIndex].x] /
                                    16.0f;
                                if (disparityDisplayed > 0) {
                                    float dist = (fx * abs(tx)) / disparityDisplayed;
                                    dev.pixelInfoZoomed[windowIndex].depth = dist;
                                }
                                else {
                                    dev.pixelInfoZoomed[windowIndex].depth = 0;
                                }
                            }
                            break;
                            default:
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

void Renderer::windowResized() {
    renderData.camera = &camera;
    renderData.deltaT = frameTimer;
    renderData.index = currentFrame;
    renderData.height = m_Height;
    renderData.width = m_Width;
    renderData.crlCamera = &cameraConnection->camPtr;

    Widgets::clear();
    // Update gui with new res
    guiManager->update((frameCounter == 0), frameTimer, renderData.width, renderData.height, &input);
    // Update general Scripts with handle to GUI
    for (auto& script : scripts) {
        if (script.second->getType() != VkRender::CRL_SCRIPT_TYPE_DISABLED)
            script.second->windowResize(&renderData, &guiManager->handles);
    }

    // Clear script and scriptnames before rebuilding
    for (const auto& scriptName : builtScriptNames) {
        pLogger->info("Deleting Script: {}", scriptName.c_str());
        scripts[scriptName].get()->onDestroyScript();
        scripts[scriptName].reset();
        scripts.erase(scriptName);
    }
    builtScriptNames.clear();

    // Scripts. Start with skybox as usual
    buildScript("Skybox");
    for (const auto& name : availableScriptNames)
        buildScript(name);

    // Recreate to fit new dimensions
    vkDestroyFramebuffer(device, selection.frameBuffer, nullptr);
    vkDestroyImage(device, selection.colorImage, nullptr);
    vkDestroyImage(device, selection.depthImage, nullptr);
    vkDestroyImageView(device, selection.colorView, nullptr);
    vkDestroyImageView(device, selection.depthView, nullptr);
    vkFreeMemory(device, selection.colorMem, nullptr);
    vkFreeMemory(device, selection.depthMem, nullptr);
    createSelectionImages();
    createSelectionFramebuffer();
    destroySelectionBuffer();
    createSelectionBuffer();
}


void Renderer::cleanUp() {
    auto startTime = std::chrono::steady_clock::now();

    usageMonitor->userEndSession();

    if (usageMonitor->hasUserLogCollectionConsent() &&
        VkRender::RendererConfig::getInstance().getUserSetting().sendUsageLogOnExit)
        usageMonitor->sendUsageLog();

    auto timeSpan = std::chrono::duration_cast<std::chrono::duration<float>>(
        std::chrono::steady_clock::now() - startTime);
    Log::Logger::getInstance()->trace("Sending logs on exit took {}s", timeSpan.count());

    for (auto& dev : guiManager->handles.devices) {
        dev.interruptConnection = true; // Disable all current connections if user wants to exit early
        cameraConnection->saveProfileAndDisconnect(&dev);
    }

    startTime = std::chrono::steady_clock::now();
    // Shutdown GUI manually since it contains thread. Not strictly necessary but nice to have
    guiManager.reset();
    timeSpan = std::chrono::duration_cast<std::chrono::duration<float>>(
        std::chrono::steady_clock::now() - startTime);
    Log::Logger::getInstance()->trace("Deleting GUI on exit took {}s", timeSpan.count());

    startTime = std::chrono::steady_clock::now();
    // Clear script and scriptnames
    for (const auto& scriptName : builtScriptNames) {
        pLogger->info("Deleting Script: {}", scriptName.c_str());
        scripts[scriptName].get()->onDestroyScript();
        scripts[scriptName].reset();
        scripts.erase(scriptName);
    }
    builtScriptNames.clear();
    destroySelectionBuffer();
    timeSpan = std::chrono::duration_cast<std::chrono::duration<float>>(
        std::chrono::steady_clock::now() - startTime);
    Log::Logger::getInstance()->trace("Deleting scripts on exit took {}s", timeSpan.count());
}


void Renderer::createSelectionFramebuffer() {
    // Depth/Stencil attachment is the same for all frame buffers
    std::array<VkImageView, 2> attachments{};
    attachments[0] = selection.colorView;
    attachments[1] = selection.depthView;
    VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(m_Width, m_Height,
        attachments.data(),
        attachments.size(),
        selection.renderPass);
    VkResult result = vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &selection.frameBuffer);
    if (result != VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");
}

void Renderer::createSelectionImages() {
    // Create picking images
    {
        // Create optimal tiled target m_Image
        VkImageCreateInfo colorImageCreateInfo = Populate::imageCreateInfo();
        colorImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        colorImageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        colorImageCreateInfo.mipLevels = 1;
        colorImageCreateInfo.arrayLayers = 1;
        colorImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        colorImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        colorImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        colorImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorImageCreateInfo.extent = {m_Width, m_Height, 1};
        colorImageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        CHECK_RESULT(vkCreateImage(device, &colorImageCreateInfo, nullptr, &selection.colorImage));
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, selection.colorImage, &memReqs);
        VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &selection.colorMem));
        CHECK_RESULT(vkBindImageMemory(device, selection.colorImage, selection.colorMem, 0));

        VkImageViewCreateInfo colorAttachmentView = {};
        colorAttachmentView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorAttachmentView.pNext = nullptr;
        colorAttachmentView.format = VK_FORMAT_R8G8B8A8_UNORM;
        colorAttachmentView.components = {
            VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_B,
            VK_COMPONENT_SWIZZLE_A
        };
        colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorAttachmentView.subresourceRange.baseMipLevel = 0;
        colorAttachmentView.subresourceRange.levelCount = 1;
        colorAttachmentView.subresourceRange.baseArrayLayer = 0;
        colorAttachmentView.subresourceRange.layerCount = 1;
        colorAttachmentView.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorAttachmentView.flags = 0;
        colorAttachmentView.image = selection.colorImage;

        VkResult result = vkCreateImageView(device, &colorAttachmentView, nullptr, &selection.colorView);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create swapchain m_Image views");

        /**DEPTH IMAGE*/
        VkImageCreateInfo imageCI{};
        imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = depthFormat;
        imageCI.extent = {m_Width, m_Height, 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        result = vkCreateImage(device, &imageCI, nullptr, &selection.depthImage);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image");

        vkGetImageMemoryRequirements(device, selection.depthImage, &memReqs);

        VkMemoryAllocateInfo memAllloc{};
        memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memAllloc.allocationSize = memReqs.size;
        memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        result = vkAllocateMemory(device, &memAllloc, nullptr, &selection.depthMem);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth m_Image memory");
        result = vkBindImageMemory(device, selection.depthImage, selection.depthMem, 0);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth m_Image memory");

        VkImageViewCreateInfo imageViewCI{};
        imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.image = selection.depthImage;
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
        result = vkCreateImageView(device, &imageViewCI, nullptr, &selection.depthView);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image m_View");
    }
}

void Renderer::createSelectionBuffer() {
    CHECK_RESULT(vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        m_Width * m_Height * 4,
        &selectionBuffer,
        &selectionMemory));

    // Create the memory backing up the buffer handle
    vkGetBufferMemoryRequirements(vulkanDevice->m_LogicalDevice, selectionBuffer, &m_MemReqs);
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = m_Width;
    bufferCopyRegion.imageExtent.height = m_Height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;
}

void Renderer::destroySelectionBuffer() {
    // Clean up staging resources
    vkFreeMemory(vulkanDevice->m_LogicalDevice, selectionMemory, nullptr);
    vkDestroyBuffer(vulkanDevice->m_LogicalDevice, selectionBuffer, nullptr);
}

void Renderer::mouseMoved(float x, float y, bool& handled) {
    float dx = mousePos.x - x;
    float dy = mousePos.y - y;

    mouseButtons.dx = dx;
    mouseButtons.dy = dy;

    bool is3DViewSelected = false;
    for (const auto& dev : guiManager->handles.devices) {
        if (dev.state != VkRender::CRL_STATE_ACTIVE)
            continue;
        is3DViewSelected = dev.selectedPreviewTab == VkRender::CRL_TAB_3D_POINT_CLOUD;
    }
    if (mouseButtons.left && guiManager->handles.info->isViewingAreaHovered &&
        is3DViewSelected) {
        // && !mouseButtons.middle) {
        camera.rotate(dx, dy);
    }
    if (mouseButtons.left && guiManager->handles.renderer3D && !guiManager->handles.info->is3DTopBarHovered)
        camera.rotate(dx, dy);

    if (mouseButtons.right) {
        if (camera.type == VkRender::Camera::arcball)
            camera.translate(glm::vec3(-dx * 0.005f, -dy * 0.005f, 0.0f));
        else
            camera.translate(-dx * 0.01f, -dy * 0.01f);
    }
    if (mouseButtons.middle && camera.type == VkRender::Camera::flycam) {
        camera.translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.0f));
    }
    else if (mouseButtons.middle && camera.type == VkRender::Camera::arcball) {
        //camera.orbitPan(static_cast<float>() -dx * 0.01f, static_cast<float>() -dy * 0.01f);
    }
    mousePos = glm::vec2(x, y);

    handled = true;
}

void Renderer::mouseScroll(float change) {
    for (const auto& item : guiManager->handles.devices) {
        if (item.state == VkRender::CRL_STATE_ACTIVE && item.selectedPreviewTab == VkRender::CRL_TAB_3D_POINT_CLOUD &&
            guiManager->handles.info->isViewingAreaHovered) {
            camera.setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
        }
    }
    if (guiManager->handles.renderer3D) {
        camera.setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
    }
}
