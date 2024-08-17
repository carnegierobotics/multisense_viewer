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

#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Core/UUID.h"

#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Populate.h"

#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/Editors/EditorDefinitions.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/Scenes/MultiSenseViewer/MultiSenseViewer.h"

namespace VkRender {
    Renderer::Renderer(const std::string &title) : VulkanRenderer(title) {
        RendererConfig &config = RendererConfig::getInstance();
        this->m_title = title;
        Log::Logger::getInstance()->setLogLevel(config.getLogLevel());
        m_logger = Log::Logger::getInstance();
        VulkanRenderer::initVulkan();
        VulkanRenderer::prepare();
        m_guiResources = std::make_shared<GuiResources>(m_vulkanDevice);
        m_logger->info("Initialized Backend");

        config.setGpuDevice(physicalDevice);
        m_usageMonitor = std::make_shared<UsageMonitor>();
        m_usageMonitor->loadSettingsFromFile();
        m_usageMonitor->userStartSession(rendererStartTime);
        // Initialize shared data across editors:

        m_sharedContextData.multiSenseRendererBridge = std::make_shared<MultiSense::MultiSenseRendererBridge>();
        m_sharedContextData.multiSenseRendererGigEVisionBridge = std::make_shared<
                MultiSense::MultiSenseRendererGigEVisionBridge>();


        VulkanRenderPassCreateInfo passCreateInfo(m_vulkanDevice, &m_allocator);
        passCreateInfo.msaaSamples = msaaSamples;
        passCreateInfo.swapchainImageCount = swapchain->imageCount;
        passCreateInfo.swapchainColorFormat = swapchain->colorFormat;
        passCreateInfo.depthFormat = depthFormat;
        passCreateInfo.height = static_cast<int32_t>(m_height);
        passCreateInfo.width = static_cast<int32_t>(m_width);

        EditorCreateInfo mainMenuEditor(m_guiResources, this, &m_sharedContextData, m_vulkanDevice, &m_allocator,
                                        m_frameBuffers.data());
        mainMenuEditor.borderSize = 0;
        mainMenuEditor.editorTypeDescription = EditorType::None;
        mainMenuEditor.resizeable = false;
        mainMenuEditor.height = static_cast<int32_t>(m_height);
        mainMenuEditor.width = static_cast<int32_t>(m_width);
        mainMenuEditor.pPassCreateInfo = passCreateInfo;

        m_mainEditor = std::make_unique<Editor>(mainMenuEditor);
        m_mainEditor->addUI("DebugWindow");
        m_mainEditor->addUI("MenuLayer");
        m_mainEditor->addUI("MainContextLayer");

        m_editorFactory = std::make_unique<EditorFactory>();

        loadEditorSettings(Utils::getMyEditorProjectConfig());

        if (m_editors.empty()) {
            // add a dummy editor to get started
            auto sizeLimits = m_mainEditor->getSizeLimits();
            EditorCreateInfo otherEditorInfo(m_guiResources, this, &m_sharedContextData, m_vulkanDevice, &m_allocator,
                                                             m_frameBuffers.data());
            otherEditorInfo.pPassCreateInfo = passCreateInfo;
            otherEditorInfo.borderSize = 5;
            otherEditorInfo.height = static_cast<int32_t>(m_height) - sizeLimits.MENU_BAR_HEIGHT; //- 100;
            otherEditorInfo.width = static_cast<int32_t>(m_width); //- 200;
            otherEditorInfo.x = 0; //+ 100;
            otherEditorInfo.y = sizeLimits.MENU_BAR_HEIGHT; //+ 050;
            otherEditorInfo.editorIndex = m_editors.size();
            otherEditorInfo.editorTypeDescription = EditorType::TestWindow;
            otherEditorInfo.uiContext = getMainUIContext();
            auto editor = createEditor(otherEditorInfo);
            m_editors.push_back(std::move(editor));
        }


        // Load scenes

        // Load the default scene
        m_scene = std::make_shared<MultiSenseViewer>(*this);
        for (auto &editor: m_editors) {
            editor->loadScene();
        }
    }

    // TODO This should actually be handled by RendererConfig. This class handles everything saving and loading config files
    void Renderer::loadEditorSettings(const std::filesystem::path &filePath) {
        std::ifstream inFile(filePath);
        if (!inFile.is_open()) {
            Log::Logger::getInstance()->error("Failed to open file for reading: {}", filePath.string());
            return;
        }

        nlohmann::json jsonContent;
        inFile >> jsonContent;
        inFile.close();

        Log::Logger::getInstance()->info("Successfully read editor settings from: {}", filePath.string());


        if (jsonContent.contains("editors")) {
            for (const auto &jsonEditor: jsonContent["editors"]) {
                EditorCreateInfo createInfo(m_guiResources, this, &m_sharedContextData, m_vulkanDevice, &m_allocator,
                                            m_frameBuffers.data());

                int32_t mainMenuBarOffset = 0;
                int32_t width = std::round(jsonEditor.value("width", 0.0) / 100 * m_width);
                int32_t height = std::round(jsonEditor.value("height", 0.0) / 100 * m_height);
                int32_t offsetX = std::round(jsonEditor.value("x", 0.0) / 100 * m_width);
                int32_t offsetY = std::round(jsonEditor.value("y", 0.0) / 100 * m_height);

                mainMenuBarOffset = offsetY == 0 ? 25 : 0;

                createInfo.x = offsetX;
                createInfo.y = offsetY;
                createInfo.y += mainMenuBarOffset;

                createInfo.width = width;
                createInfo.height = height - mainMenuBarOffset;

                if (createInfo.width + createInfo.x > m_width) {
                    createInfo.width = m_width - createInfo.x;
                }

                if (createInfo.height + createInfo.y > m_height) {
                    createInfo.height = m_height - createInfo.y;
                }

                createInfo.borderSize = jsonEditor.value("borderSize", 5);
                createInfo.editorTypeDescription = stringToEditorType(jsonEditor.value("editorTypeDescription", ""));
                createInfo.resizeable = jsonEditor.value("resizeable", true);
                createInfo.editorIndex = jsonEditor.value("editorIndex", 0);
                createInfo.uiContext = getMainUIContext();

                VulkanRenderPassCreateInfo passCreateInfo(m_vulkanDevice, &m_allocator);
                passCreateInfo.msaaSamples = msaaSamples;
                passCreateInfo.swapchainImageCount = swapchain->imageCount;
                passCreateInfo.swapchainColorFormat = swapchain->colorFormat;
                passCreateInfo.depthFormat = depthFormat;
                passCreateInfo.height = static_cast<int32_t>(m_height);
                passCreateInfo.width = static_cast<int32_t>(m_width);
                createInfo.pPassCreateInfo = passCreateInfo;
                if (jsonEditor.contains("uiLayers") && jsonEditor["uiLayers"].is_array()) {
                    createInfo.uiLayers = jsonEditor["uiLayers"].get<std::vector<std::string> >();
                } else {
                    createInfo.uiLayers.clear();
                }
                // Create an Editor object with the createInfo
                m_editors.push_back(std::move(createEditor(createInfo)));

                Log::Logger::getInstance()->info("Loaded editor {}: type = {}, x = {}, y = {}, width = {}, height = {}",
                                                 createInfo.editorIndex,
                                                 editorTypeToString(createInfo.editorTypeDescription), createInfo.x,
                                                 createInfo.y, createInfo.width, createInfo.height);
            }
        }

    }


    void Renderer::loadScene(std::filesystem::path scenePath) {

        if (scenePath == "Default Scene")
            m_scene = (std::make_shared<DefaultScene>(*this));
        else {
            m_scene = (std::make_shared<MultiSenseViewer>(*this));
        }
        for (auto &editor: m_editors) {
            editor->loadScene();
        }
    }

    void Renderer::deleteScene(std::filesystem::path scenePath) {
        // Find the scene to delete
        Log::Logger::getInstance()->info("Deleting Scene with Reference count: {}", m_scene.use_count());
        //m_scene.reset();

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

    void Renderer::updateUniformBuffers() {
        // update imgui io:

        ImGui::SetCurrentContext(m_mainEditor->guiContext());
        ImGuiIO &mainIO = ImGui::GetIO();
        mainIO.DeltaTime = frameTimer;
        mainIO.WantCaptureMouse = true;
        mainIO.MousePos = ImVec2(mouse.x, mouse.y);
        mainIO.MouseDown[0] = mouse.left;
        mainIO.MouseDown[1] = mouse.right;
        for (auto &editor: m_editors) {
            ImGui::SetCurrentContext(editor->guiContext());
            ImGuiIO &otherIO = ImGui::GetIO();
            otherIO.DeltaTime = frameTimer;
            otherIO.WantCaptureMouse = true;
            otherIO.MousePos = ImVec2(mouse.x - editor->getCreateInfo().x, mouse.y - editor->getCreateInfo().y);
            otherIO.MouseDown[0] = mouse.left;
            otherIO.MouseDown[1] = mouse.right;
        }
        updateEditors();
        m_mainEditor->update((frameCounter == 0), frameTimer, &input);
        m_logger->frameNumber = frameID;
        m_sharedContextData.multiSenseRendererBridge->update();

        // TODO reconsider if we should call crl updates here?

        std::string versionRemote;
        /*
        if (m_mainEditor->handles.askUserForNewVersion && m_usageMonitor->getLatestAppVersionRemote(&versionRemote)) {
            std::string localAppVersion = RendererConfig::getInstance().getAppVersion();
            Log::Logger::getInstance()->info("New Version is Available: Local version={}, available version={}",
                                             localAppVersion, versionRemote);
            m_mainEditor->handles.newVersionAvailable = Utils::isLocalVersionLess(localAppVersion, versionRemote);
        }
         */



        //if (keyPress == GLFW_KEY_SPACE) {
        //    m_cameras[m_selectedCameraTag].resetPosition();
        //}
        /**@brief Record commandbuffers for obj models
        // Accessing components in a non-copying manner

        for (auto entity: m_registry.view<DefaultGraphicsPipelineComponent2>()) {
            auto &resources = m_registry.get<DefaultGraphicsPipelineComponent2>(entity);
            const auto &transform = m_registry.get<TransformComponent>(entity);
            const auto &currentCamera = m_cameras[m_selectedCameraTag];
            resources.updateTransform(transform);
            resources.updateView(currentCamera);
            resources.update(currentFrame);
        }
        /**@brief Record commandbuffers for obj models
        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<CustomModelComponent>()) {
            auto &resources = m_registry.get<CustomModelComponent>(entity);
            const auto &transform = m_registry.get<TransformComponent>(entity);
            const auto &currentCamera = m_cameras[m_selectedCameraTag];
            resources.updateTransform(transform);
            resources.updateView(currentCamera);
            resources.update(currentFrame);
        }
        /**@brief Record commandbuffers for obj models
        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<CameraGraphicsPipelineComponent>()) {
            auto &resources = m_registry.get<CameraGraphicsPipelineComponent>(entity);
            auto &tag = m_registry.get<TagComponent>(entity);
            const auto &transform = m_registry.get<TransformComponent>(entity);
            const auto &currentCamera = m_cameras[m_selectedCameraTag];
            resources.updateTransform(transform);
            resources.updateView(currentCamera);
            resources.update(currentFrame);
        } /**@brief Record commandbuffers for gltf models
        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<DefaultPBRGraphicsPipelineComponent>()) {
            auto &resources = m_registry.get<DefaultPBRGraphicsPipelineComponent>(entity);
            auto &tag = m_registry.get<TagComponent>(entity);
            const auto &transform = m_registry.get<TransformComponent>(entity);
            const auto &currentCamera = m_cameras[m_selectedCameraTag];
            resources.updateTransform(transform);
            resources.updateView(currentCamera);
            resources.update(currentFrame);
        }

        /*
        // Update GUI
        m_guiManager->handles.info->frameID = frameID;
        m_guiManager->handles.info->applicationRuntime = runTime;
        m_guiManager->update((frameCounter == 0), frameTimer, m_renderUtils.width, m_renderUtils.height, &input);


        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->uiUpdate(&m_guiManager->handles);
        }


        // Load components from UI actions?
        // Load new obj file
        if (m_guiManager->handles.m_paths.updateObjPath) {
            Log::Logger::getInstance()->info("Loading new model from {}",
                                             m_guiManager->handles.m_paths.importFilePath.string());
            std::filesystem::path filename = m_guiManager->handles.m_paths.importFilePath.filename();

            auto entity = createEntity(filename.replace_extension().string());
            auto &component = entity.addComponent<OBJModelComponent>(m_guiManager->handles.m_paths.importFilePath,
                                                                     m_vulkanDevice);

            entity.addComponent<DefaultGraphicsPipelineComponent2>(&m_renderUtils).bind(component);
            entity.addComponent<DepthRenderPassComponent>();
        }
        // Load new gltf file
        if (m_guiManager->handles.m_paths.updateGLTFPath) {
            Log::Logger::getInstance()->info("Loading new model from {}",
                                             m_guiManager->handles.m_paths.importFilePath.string());
            std::filesystem::path filename = m_guiManager->handles.m_paths.importFilePath.filename();
            auto entity = createEntity(filename.replace_extension().string());
            auto &component = entity.addComponent<VkRender::GLTFModelComponent>(
                    m_guiManager->handles.m_paths.importFilePath.string(),
                    m_vulkanDevice);
            auto &sky = findEntityByName("Skybox").getComponent<VkRender::SkyboxGraphicsPipelineComponent>();
            entity.addComponent<VkRender::DefaultPBRGraphicsPipelineComponent>(&m_renderUtils, component, sky);
            entity.addComponent<DepthRenderPassComponent>();

        }

        // Update camera gizmos
        for (auto entity: m_registry.view<CameraGraphicsPipelineComponent>()) {
            auto &resources = m_registry.get<CameraGraphicsPipelineComponent>(entity);
            const auto *camera = m_registry.get<CameraComponent>(entity).camera;
            auto &transform = m_registry.get<TransformComponent>(entity);
            transform.setQuaternion(camera->pose.q);
            transform.setPosition(camera->pose.pos);
            const auto &currentCamera = m_cameras[m_selectedCameraTag];
            resources.updateTransform(transform);
            resources.updateView(currentCamera);
            resources.update(currentFrame);
        }
        */
    }

    std::unique_ptr<Editor> Renderer::createEditor(EditorCreateInfo &createInfo) {
        auto editor = createEditorWithUUID(UUID(), createInfo);
        return editor;
    }

    std::unique_ptr<Editor> Renderer::createEditorWithUUID(UUID uuid, EditorCreateInfo &createInfo) {
        return m_editorFactory->createEditor(createInfo.editorTypeDescription, createInfo, uuid);
    }

    void Renderer::recreateEditor(std::unique_ptr<Editor> &editor, EditorCreateInfo &createInfo) {
        auto newEditor = createEditor(createInfo);
        newEditor->ui() = editor->ui();
        newEditor->onSceneLoad();
        editor = std::move(newEditor);
    }

    void Renderer::updateEditors() {
        // Reorder Editors elements according to UI
        for (auto &editor: m_editors) {
            if (editor->ui().changed) {
                // Set a new one
                Log::Logger::getInstance()->info("New Editor requested");
                auto &ci = editor->getCreateInfo();
                ci.editorTypeDescription = editor->ui().selectedType;
                recreateEditor(editor, ci);
            }
        }
        handleEditorResize();
    }

    void Renderer::onRender() {
        /** Generate Draw Commands **/
        for (auto &editor: m_editors) {
            editor->render(drawCmdBuffers);
        }
        m_mainEditor->render(drawCmdBuffers);
        /** IF WE SHOULD RENDER SECOND IMAGE FOR MOUSE PICKING EVENTS (Reason: let user see PerPixelInformation)
         *  THIS INCLUDES RENDERING SELECTED OBJECTS AND COPYING CONTENTS BACK TO CPU INSTEAD OF DISPLAYING TO SCREEN **/
    }

    void Renderer::windowResized(int32_t dx, int32_t dy, double widthScale, double heightScale) {

        Widgets::clear();

        if (dx != 0)
            Editor::windowResizeEditorsHorizontal(dx, widthScale, m_editors, m_width);
        if (dy != 0)
            Editor::windowResizeEditorsVertical(dy, heightScale, m_editors, m_height);

        for (auto &editor: m_editors) {
            auto &ci = editor->getCreateInfo();
            editor->resize(ci);
        }
        auto &ci = m_mainEditor->getCreateInfo();
        ci.width = m_width;
        ci.height = m_height;
        ci.frameBuffers = m_frameBuffers.data();
        m_mainEditor->resize(ci);
    }

    void Renderer::cleanUp() {
        if (std::filesystem::exists(Utils::getRuntimeConfigFilePath())) {
            std::filesystem::remove(Utils::getRuntimeConfigFilePath());
            Log::Logger::getInstance()->info("Removed runtime config file before cleanup {}",
                                             Utils::getRuntimeConfigFilePath().string().c_str());
        }

        auto startTime = std::chrono::steady_clock::now();

        m_usageMonitor->userEndSession();
        RendererConfig::getInstance().getUserSetting().applicationWidth = m_width;
        RendererConfig::getInstance().getUserSetting().applicationHeight = m_height;

        if (m_usageMonitor->hasUserLogCollectionConsent() &&
            RendererConfig::getInstance().getUserSetting().sendUsageLogOnExit)
            m_usageMonitor->sendUsageLog();

        RendererConfig::getInstance().saveSettings(this);
        auto timeSpan = std::chrono::duration_cast<std::chrono::duration<float> >(
                std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Sending logs on exit took {}s", timeSpan.count());

        startTime = std::chrono::steady_clock::now();
        // Shutdown GUI manually since it contains thread. Not strictly necessary but nice to have
        //m_guiManager.reset();
        timeSpan = std::chrono::duration_cast<std::chrono::duration<float> >(
                std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Deleting GUI on exit took {}s", timeSpan.count());
        startTime = std::chrono::steady_clock::now();


        timeSpan = std::chrono::duration_cast<std::chrono::duration<float> >(
                std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Deleting entities on exit took {}s", timeSpan.count());
        // Destroy framebuffer
    }

    void Renderer::handleEditorResize() {
        //// UPDATE EDITOR WITH UI EVENTS - Very little logic here
        for (auto &editor: m_editors) {
            Editor::handleHoverState(editor, mouse);
            Editor::handleClickState(editor, mouse);
        }

        bool showHandCursor = false;
        bool showCrosshairCursor = false;
        bool anyCornerClicked = false;
        bool anyResizeHovered = false;
        bool horizontalResizeHovered = false;
        for (auto &editor: m_editors) {
            if (editor->ui().cornerBottomLeftHovered && editor->getCreateInfo().resizeable) showHandCursor = true;
            if (editor->ui().cornerBottomLeftClicked && editor->getCreateInfo().resizeable) showCrosshairCursor = true;

            if (editor->ui().cornerBottomLeftClicked) anyCornerClicked = true;
            if (editor->ui().resizeHovered) anyResizeHovered = true;
            if (EditorBorderState::Left == editor->ui().lastHoveredBorderType ||
                EditorBorderState::Right == editor->ui().lastHoveredBorderType)
                horizontalResizeHovered = true;
        }

        //// UPDATE EDITOR Based on UI EVENTS
        for (auto &editor: m_editors) {
            // Dont update the editor if managed by another instance
            if (!editor->ui().indirectlyActivated) {
                Editor::handleIndirectClickState(m_editors, editor, mouse);
            }
            Editor::handleDragState(editor, mouse);
        }

        Editor::checkIfEditorsShouldMerge(m_editors);

        resizeEditors(anyCornerClicked);
        for (auto &editor: m_editors) {
            editor->update((frameCounter == 0), frameTimer, &input);
            if (!mouse.left) {
                if (editor->ui().indirectlyActivated) {
                    Editor::handleHoverState(editor, mouse);
                    editor->ui().lastClickedBorderType = None;
                    editor->ui().active = false;
                    editor->ui().indirectlyActivated = false;
                }
                editor->ui().resizeActive = false;
                editor->ui().cornerBottomLeftClicked = false;
                editor->ui().dragHorizontal = false;
                editor->ui().dragVertical = false;
                editor->ui().dragActive = false;
                editor->ui().splitting = false;
                editor->ui().lastPressedPos = glm::ivec2(-1, -1);
                editor->ui().dragDelta = glm::ivec2(0, 0);
                editor->ui().cursorDelta = glm::ivec2(0, 0);
            }
            if (!mouse.right) {
                editor->ui().rightClickBorder = false;
                editor->ui().lastRightClickedBorderType = None;
            }
            if (mouse.left && mouse.action == GLFW_PRESS) {
                Log::Logger::getInstance()->info("We Left-clicked Editor: {}'s area :{}",
                                                 editor->getCreateInfo().editorIndex,
                                                 editor->ui().lastClickedBorderType);
            }
            if (mouse.right && mouse.action == GLFW_PRESS) {
                Log::Logger::getInstance()->info("We Right-clicked Editor: {}'s area :{}. Merge? {}",
                                                 editor->getCreateInfo().editorIndex,
                                                 editor->ui().lastClickedBorderType,
                                                 editor->ui().rightClickBorder);
            }
        }
        bool splitEditors = false;
        uint32_t splitEditorIndex = UINT32_MAX;
        for (size_t index = 0; auto &editor: m_editors) {
            if (editor->getCreateInfo().resizeable &&
                editor->ui().cornerBottomLeftClicked &&
                editor->getCreateInfo().width > 100 && editor->getCreateInfo().height > 100 &&
                (editor->ui().dragHorizontal || editor->ui().dragVertical) &&
                !editor->ui().splitting) {
                splitEditors = true;
                splitEditorIndex = index;
            }
            index++;
        }
        if (splitEditors) {
            splitEditor(splitEditorIndex);
        }

        bool mergeEditor = false;
        std::array<UUID, 2> editorsUUID;
        for (size_t index = 0; auto &editor: m_editors) {
            if (editor->ui().shouldMerge) {
                editorsUUID[index] = editor->getUUID();
                index++;
                mergeEditor = true;
            }
        }
        if (mergeEditor)
            mergeEditors(editorsUUID);
        //
        if (showCrosshairCursor) {
            glfwSetCursor(window, m_cursors.crossHair);
        } else if (showHandCursor) {
            glfwSetCursor(window, m_cursors.hand);
        } else if (anyResizeHovered) {
            glfwSetCursor(window, horizontalResizeHovered ? m_cursors.resizeHorizontal : m_cursors.resizeVertical);
        } else {
            glfwSetCursor(window, m_cursors.arrow);
        }
    }


    void Renderer::resizeEditors(bool anyCornerClicked) {
        bool isValidResizeAll = true;
        for (auto &editor: m_editors) {
            if (editor->ui().resizeActive && (!anyCornerClicked || editor->ui().splitting)) {
                auto createInfo = getNewEditorCreateInfo(editor);
                if (!Editor::isValidResize(createInfo, editor))
                    isValidResizeAll = false;
            }
        }
        if (isValidResizeAll) {
            for (auto &editor: m_editors) {
                if (editor->ui().resizeActive && (!anyCornerClicked || editor->ui().splitting)) {
                    auto createInfo = getNewEditorCreateInfo(editor);
                    editor->resize(createInfo);
                }
            }
        }
    }

    void Renderer::splitEditor(uint32_t splitEditorIndex) {
        auto &editor = m_editors[splitEditorIndex];
        EditorCreateInfo &editorCreateInfo = editor->getCreateInfo();
        EditorCreateInfo newEditorCreateInfo(m_guiResources, this, &m_sharedContextData, m_vulkanDevice, &m_allocator,
                                             m_frameBuffers.data());

        EditorCreateInfo::copy(&newEditorCreateInfo, &editorCreateInfo);

        if (editor->ui().dragHorizontal) {
            editorCreateInfo.width -= editor->ui().dragDelta.x;
            editorCreateInfo.x += editor->ui().dragDelta.x;
            newEditorCreateInfo.width = editor->ui().dragDelta.x;
        } else {
            editorCreateInfo.height += editor->ui().dragDelta.y;
            newEditorCreateInfo.height = -editor->ui().dragDelta.y;
            newEditorCreateInfo.y = editorCreateInfo.height + editorCreateInfo.y;
        }
        newEditorCreateInfo.editorIndex = m_editors.size();
        if (!editor->validateEditorSize(newEditorCreateInfo) ||
            !editor->validateEditorSize(editorCreateInfo))
            return;
        editor->resize(editorCreateInfo);
        auto newEditor = createEditor(newEditorCreateInfo);
        editor->ui().resizeActive = true;
        newEditor->ui().resizeActive = true;
        editor->ui().active = true;
        newEditor->ui().active = true;
        editor->ui().splitting = true;
        newEditor->ui().splitting = true;
        if (editor->ui().dragHorizontal) {
            editor->ui().lastClickedBorderType = EditorBorderState::Left;
            newEditor->ui().lastClickedBorderType = EditorBorderState::Right;
        } else {
            editor->ui().lastClickedBorderType = EditorBorderState::Bottom;
            newEditor->ui().lastClickedBorderType = EditorBorderState::Top;
        }

        newEditor->onSceneLoad();
        m_editors.push_back(std::move(newEditor));
    }

    void Renderer::mergeEditors(const std::array<UUID, 2> &mergeEditorIndices) {
        UUID id1 = mergeEditorIndices[0];
        UUID id2 = mergeEditorIndices[1];

        auto &editor1 = findEditorByUUID(id1);
        auto &editor2 = findEditorByUUID(id2);

        if (!editor1 || !editor2) {
            Log::Logger::getInstance()->info("Wanted to merge editors: {} and {} but they were not found",
                                             id1.operator std::string(), id2.operator std::string());
            return;
        }
        // Implement your merging logic here
        // For example, combine editor2's properties into editor1
        // editor1.someProperty += editor2.someProperty;
        editor1->ui().shouldMerge = false;
        editor2->ui().shouldMerge = false;

        auto &ci1 = editor1->getCreateInfo();
        auto &ci2 = editor2->getCreateInfo();

        int32_t newX = std::min(ci1.x, ci2.x);
        int32_t newY = std::min(ci1.y, ci2.y);
        int32_t newWidth = ci1.width + ci2.width;
        int32_t newHeight = ci1.height + ci2.height;

        if (editor1->ui().lastRightClickedBorderType & EditorBorderState::HorizontalBorders) {
            ci1.height = newHeight;
        } else if (editor1->ui().lastRightClickedBorderType & EditorBorderState::VerticalBorders) {
            ci1.width = newWidth;
        }
        ci1.x = newX;
        ci1.y = newY;

        Log::Logger::getInstance()->info("Merging editor {} into editor {}.", editor2->getCreateInfo().editorIndex,
                                         editor1->getCreateInfo().editorIndex);

        auto editor2UUID = editor2->getUUID();
        // Remove editor2 safely based on UUID
        m_editors.erase(
                std::remove_if(m_editors.begin(), m_editors.end(),
                               [editor2UUID](const std::unique_ptr<Editor> &editor) {
                                   return editor->getUUID() == editor2UUID;
                               }),
                m_editors.end()
        );
        editor1->resize(ci1);
    }

    EditorCreateInfo Renderer::getNewEditorCreateInfo(std::unique_ptr<Editor> &editor) {
        EditorCreateInfo newEditorCI(m_guiResources, this, &m_sharedContextData, m_vulkanDevice, &m_allocator,
                                     m_frameBuffers.data());
        EditorCreateInfo::copy(&newEditorCI, &editor->getCreateInfo());

        switch (editor->ui().lastClickedBorderType) {
            case EditorBorderState::Left:
                newEditorCI.x = editor->ui().x + editor->ui().cursorDelta.x;
                newEditorCI.width = editor->ui().width - editor->ui().cursorDelta.x;
                break;
            case EditorBorderState::Right:
                newEditorCI.width = editor->ui().width + editor->ui().cursorDelta.x;
                break;
            case EditorBorderState::Top:
                newEditorCI.y = editor->ui().y + editor->ui().cursorDelta.y;
                newEditorCI.height = editor->ui().height - editor->ui().cursorDelta.y;
                break;
            case EditorBorderState::Bottom:
                newEditorCI.height = editor->ui().height + editor->ui().cursorDelta.y;
                break;
            default:
                Log::Logger::getInstance()->trace(
                        "Resize is somehow active but we have not clicked any borders: {}",
                        editor->getCreateInfo().editorIndex);
                break;
        }
        return newEditorCI;
    }

    void Renderer::mouseMoved(float x, float y, bool &handled) {
        mouse.insideApp = !(x < 0 || x > m_width || y < 0 || y > m_height);
        float dx = x - mouse.x;
        float dy = y - mouse.y;
        mouse.dx += dx;
        mouse.dy += dy;
        Log::Logger::getInstance()->trace("Cursor velocity: ({},{}), pos: ({},{})", mouse.dx, mouse.dy, mouse.x, mouse.y);

        for (auto& editor : m_editors){
                editor->onMouseMove(mouse);
        }
        // UPdate camera if we have one selected
        if (!m_selectedCameraTag.empty()) {
            /*
            auto it = m_cameras.find(m_selectedCameraTag);
            if (it != m_cameras.end()) {
                if (mouse.left) {
                    // && !mouseButtons.middle) {
                    m_cameras[m_selectedCameraTag].rotate(dx, dy);
                }
                //if (mouseButtons.left && m_guiManager->handles.renderer3D)
                //    m_cameras[m_selectedCameraTag].rotate(dx, dy);
                if (mouse.right) {
                    if (m_cameras[m_selectedCameraTag].m_type == Camera::arcball)
                        m_cameras[m_selectedCameraTag].translate(glm::vec3(-dx * 0.005f, -dy * 0.005f, 0.0f));
                    else
                        m_cameras[m_selectedCameraTag].translate(-dx * 0.01f, -dy * 0.01f);
                }
                if (mouse.middle && m_cameras[m_selectedCameraTag].m_type == Camera::flycam) {
                    m_cameras[m_selectedCameraTag].translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.0f));
                } else if (mouse.middle && m_cameras[m_selectedCameraTag].m_type == Camera::arcball) {
                    //camera.orbitPan(static_cast<float>() -dx * 0.01f, static_cast<float>() -dy * 0.01f);
                }
            }
             */
        }
        mouse.x = x;
        mouse.y = y;
        handled = true;
    }

    void Renderer::mouseScroll(float change) {
        ImGuiIO &io = ImGui::GetIO();
        io.MouseWheel += 0.5f * static_cast<float>(change);
        for (auto& editor : m_editors){
            editor->onMouseScroll(change);
        }
        /*
        if (m_guiManager->handles.renderer3D) {
            m_cameras[m_selectedCameraTag].setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
        }
         */
    }


    std::unique_ptr<Editor> &Renderer::findEditorByUUID(const UUID &uuid) {
        for (auto &editor: m_editors) {
            if (uuid == editor->getUUID()) {
                return editor;
            }
        } // TODO what to return?
    }


    void Renderer::postRenderActions() {
        // Reset mousewheel across imgui contexts
        /*
        for (std::vector<ImGuiContext *> list = {m_mainEditor->m_guiManager->m_imguiContext,
                                                 m_editors[0].m_guiManager->m_imguiContext}; auto &ctx : list) {
            ImGui::SetCurrentContext(ctx);
            ImGuiIO &io = ImGui::GetIO();
            io.MouseWheel = 0;
        }
        */
    }

    /*

           */

    void Renderer::keyboardCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
        for (auto &editor: m_editors) {
            editor->onKeyCallback(input);
        }
        //m_cameras[m_selectedCameraTag].keys.up = input.keys.up;
        //m_cameras[m_selectedCameraTag].keys.down = input.keys.down;
        //m_cameras[m_selectedCameraTag].keys.left = input.keys.left;
        //m_cameras[m_selectedCameraTag].keys.right = input.keys.right;
    }

    std::shared_ptr<Scene> Renderer::activeScene() {
        return m_scene;
    }

};
