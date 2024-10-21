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

#include "Application.h"

#include "Viewer/Scenes/SceneSerializer.h"
#include "Viewer/VkRender/Core/UUID.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/VkRender/Editors/EditorDefinitions.h"

namespace VkRender {
    Application::Application(const std::string &title) : VulkanRenderer(title) {
        ApplicationConfig &config = ApplicationConfig::getInstance();
        this->m_title = title;
        Log::Logger::getInstance()->setLogLevel(config.getLogLevel());
        VulkanRenderer::initVulkan();
        VulkanRenderer::prepare();
        Log::Logger::getInstance()->info("Initialized Backend");
        config.setGpuDevice(physicalDevice);

        m_usageMonitor = std::make_shared<UsageMonitor>();
        m_usageMonitor->userStartSession(rendererStartTime);


        m_guiResources = std::make_shared<GuiAssets>(this);

        auto &userSetting = ApplicationConfig::getInstance().getUserSetting();
        // Create a scene and load deserialize from file if a file exsits
        std::shared_ptr<Scene> scene = newScene();
        if (std::filesystem::exists(userSetting.lastActiveScenePath)) {
            SceneSerializer serializer(scene);
            serializer.deserialize(userSetting.lastActiveScenePath);
        }

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


        std::string lastActiveProject = ApplicationConfig::getInstance().getUserSetting().projectName;
        loadProject(Utils::getProjectFileFromName(lastActiveProject));

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

        m_multiSense = std::make_shared<MultiSense::MultiSenseRendererBridge>();
        m_multiSense->setup();
    }

    Editor *Application::getSceneRendererByUUID(const UUID &uuid) {
        if (m_sceneRenderers.contains(uuid))
            return m_sceneRenderers.find(uuid)->second.get();
        return nullptr;
    }

    Editor *Application::addSceneRendererWithUUID(const UUID &uuid) {
        EditorCreateInfo sceneRenderer(m_guiResources, this, &m_sharedContextData, m_vulkanDevice, &m_allocator,
                                       m_frameBuffers.data());
        VulkanRenderPassCreateInfo passCreateInfo(m_vulkanDevice, &m_allocator);
        passCreateInfo.msaaSamples = msaaSamples;
        passCreateInfo.swapchainImageCount = swapchain->imageCount;
        passCreateInfo.swapchainColorFormat = swapchain->colorFormat;
        passCreateInfo.depthFormat = depthFormat;
        passCreateInfo.height = static_cast<int32_t>(m_height);
        passCreateInfo.width = static_cast<int32_t>(m_width);
        sceneRenderer.borderSize = 0;
        sceneRenderer.resizeable = false;
        sceneRenderer.height = static_cast<int32_t>(m_height);
        sceneRenderer.width = static_cast<int32_t>(m_width);
        sceneRenderer.pPassCreateInfo = passCreateInfo;
        sceneRenderer.editorTypeDescription = EditorType::SceneRenderer;

        m_sceneRenderers[uuid] = createEditorWithUUID(uuid, sceneRenderer);
        m_sceneRenderers[uuid]->loadScene(m_activeScene);
        return m_sceneRenderers[uuid].get();
    }

    void Application::loadProject(const std::filesystem::path &filePath) {
        // Remove project if already loaded
        m_editors.clear();
        m_projectConfig = {};
        // Then load from file
        std::ifstream inFile(filePath);
        if (!inFile.is_open()) {
            Log::Logger::getInstance()->error("Failed to open file for reading: {}", filePath.string());
            return;
        }
        nlohmann::json jsonContent;
        inFile >> jsonContent;
        inFile.close();

        Log::Logger::getInstance()->info("Successfully read editor settings from: {}", filePath.string());
        if (jsonContent.contains("generalSettings")) {
            const auto &jsonGeneralSettings = jsonContent["generalSettings"];
            m_projectConfig.name = jsonGeneralSettings.value("projectName", "MultiSense Editor");
            m_projectConfig.editorTypes = getAllEditorTypes();
        }

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
                createInfo.editorTypeDescription = stringToEditorType(
                        jsonEditor.value("editorTypeDescription", ""));
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
                Log::Logger::getInstance()->info(
                        "Loaded editor {}: type = {}, x = {}, y = {}, width = {}, height = {}",
                        createInfo.editorIndex,
                        editorTypeToString(createInfo.editorTypeDescription), createInfo.x,
                        createInfo.y, createInfo.width, createInfo.height);
            }
        }

        // Then load scenes after editors has been loaded
        if (jsonContent.contains("generalSettings")) {
            const auto &jsonGeneralSettings = jsonContent["generalSettings"];
            std::string projectScene = jsonGeneralSettings.value("sceneName", "Default Scene");
            loadScene(projectScene);
        }
    }

    // TODO make scene objects serializeable and loadable.
    void Application::loadScene(const std::filesystem::path &scenePath) {
        /*
        for (auto &editor: m_sceneRenderers) {
            editor.second->loadScene(m_activeScene);
        }
        */
        for (auto &editor: m_editors) {
            editor->loadScene(std::shared_ptr<Scene>(m_activeScene));
        }
    }

    void Application::deleteScene(std::filesystem::path scenePath) {
        // Find the scene to delete
        Log::Logger::getInstance()->info("Deleting Scene with Reference count: {}", m_activeScene.use_count());
        //m_scene.reset();
    }


    void Application::addDeviceFeatures() {
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

    void Application::updateUniformBuffers() {
        // update imgui io:
        Log::Logger::getInstance()->frameNumber = frameID;

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

        for (auto &editor: m_sceneRenderers) {
            editor.second->update();
        }
        updateEditors();
        m_mainEditor->update();

        m_multiSense->update();
    }

    std::unique_ptr<Editor> Application::createEditor(EditorCreateInfo &createInfo) {
        auto editor = createEditorWithUUID(UUID(), createInfo);
        return editor;
    }

    std::unique_ptr<Editor> Application::createEditorWithUUID(UUID uuid, EditorCreateInfo &createInfo) {
        return m_editorFactory->createEditor(createInfo.editorTypeDescription, createInfo, uuid);
    }

    void Application::recreateEditor(std::unique_ptr<Editor> &editor, EditorCreateInfo &createInfo) {
        auto newEditor = createEditor(createInfo);
        newEditor->ui() = editor->ui();
        newEditor->onSceneLoad(std::shared_ptr<Scene>());
        editor = std::move(newEditor);
    }

    void Application::updateEditors() {
        // Reorder Editors elements according to UI
        for (auto &editor: m_editors) {
            if (editor->ui()->changed) {
                // Set a new one
                Log::Logger::getInstance()->info("New Editor requested");
                auto &ci = editor->getCreateInfo();
                ci.editorTypeDescription = editor->ui()->selectedType;
                recreateEditor(editor, ci);
            }

            editor->update();
        }
        handleEditorResize();

        // Update shared data
        for (auto &editor: m_editors) {
            m_sharedEditorData.selectedUUIDContext = std::make_shared<UUID>(editor->getUUID());
        }
    }

    void Application::onRender() {
        /** Generate Draw Commands **/

        for (auto &editor: m_sceneRenderers) {
            editor.second->render(drawCmdBuffers);
        }

        for (auto &editor: m_editors) {
            editor->render(drawCmdBuffers);
        }

        m_mainEditor->render(drawCmdBuffers);
    }

    void Application::windowResized(int32_t dx, int32_t dy, double widthScale, double heightScale) {
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

        for (auto &editor: m_sceneRenderers) {
            auto &ci = editor.second->getCreateInfo();
            editor.second->resize(ci);
        }


        m_mainEditor->resize(ci);
    }

    void Application::cleanUp() {
        auto &userSetting = ApplicationConfig::getInstance().getUserSetting();
        userSetting.projectName = m_projectConfig.name;
        m_usageMonitor->userEndSession();
        ApplicationConfig::getInstance().saveSettings();
        m_activeScene->deleteAllEntities();
        m_sceneRenderers.clear();
    }

    void Application::handleEditorResize() {
        //// UPDATE EDITOR WITH UI EVENTS - Very little logic here
        for (auto &editor: m_editors) {
            Editor::handleHoverState(editor, mouse);
            Editor::handleClickState(editor, mouse);
        }

        bool hoveredLeftCornerResizeable = false;
        bool clickedLeftCornerResizeable = false;
        bool anyCornerClicked = false;
        bool anyResizeHovered = false;
        bool horizontalResizeHovered = false;
        for (auto &editor: m_editors) {
            if (editor->ui()->cornerBottomLeftHovered && editor->getCreateInfo().resizeable)
                hoveredLeftCornerResizeable = true;
            if (editor->ui()->cornerBottomLeftClicked && editor->getCreateInfo().resizeable)
                clickedLeftCornerResizeable = true;

            if (editor->ui()->cornerBottomLeftClicked) anyCornerClicked = true;
            if (editor->ui()->resizeHovered) anyResizeHovered = true;
            if (EditorBorderState::Left == editor->ui()->lastHoveredBorderType ||
                EditorBorderState::Right == editor->ui()->lastHoveredBorderType)
                horizontalResizeHovered = true;
        }

        //// UPDATE EDITOR Based on UI EVENTS
        for (auto &editor: m_editors) {
            // Dont update the editor if managed by another instance
            if (!editor->ui()->indirectlyActivated) {
                Editor::handleIndirectClickState(m_editors, editor, mouse);
            }
            Editor::handleDragState(editor, mouse);
        }

        Editor::checkIfEditorsShouldMerge(m_editors);

        resizeEditors(anyCornerClicked);
        for (auto &editor: m_editors) {
            if (!mouse.left) {
                if (editor->ui()->indirectlyActivated) {
                    Editor::handleHoverState(editor, mouse);
                    editor->ui()->lastClickedBorderType = None;
                    editor->ui()->active = false;
                    editor->ui()->indirectlyActivated = false;
                }
                editor->ui()->resizeActive = false;
                editor->ui()->cornerBottomLeftClicked = false;
                editor->ui()->dragHorizontal = false;
                editor->ui()->dragVertical = false;
                editor->ui()->dragActive = false;
                editor->ui()->splitting = false;
                editor->ui()->lastPressedPos = glm::ivec2(-1, -1);
                editor->ui()->dragDelta = glm::ivec2(0, 0);
                editor->ui()->cursorDelta = glm::ivec2(0, 0);
            }
            if (!mouse.right) {
                editor->ui()->rightClickBorder = false;
                editor->ui()->lastRightClickedBorderType = None;
            }
            if (mouse.left && mouse.action == GLFW_PRESS) {
                Log::Logger::getInstance()->trace("We Left-clicked Editor: {}'s area :{}",
                                                  editor->getCreateInfo().editorIndex,
                                                  editor->ui()->lastClickedBorderType);
            }
            if (mouse.right && mouse.action == GLFW_PRESS) {
                Log::Logger::getInstance()->trace("We Right-clicked Editor: {}'s area :{}. Merge? {}",
                                                  editor->getCreateInfo().editorIndex,
                                                  editor->ui()->lastClickedBorderType,
                                                  editor->ui()->rightClickBorder);
            }
        }
        bool splitEditors = false;
        uint32_t splitEditorIndex = UINT32_MAX;
        for (size_t index = 0; auto &editor: m_editors) {
            if (editor->getCreateInfo().resizeable &&
                editor->ui()->cornerBottomLeftClicked &&
                editor->getCreateInfo().width > 100 && editor->getCreateInfo().height > 100 &&
                (editor->ui()->dragHorizontal || editor->ui()->dragVertical) &&
                !editor->ui()->splitting) {
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
            if (editor->ui()->shouldMerge) {
                editorsUUID[index] = editor->getUUID();
                index++;
                mergeEditor = true;
            }
        }
        if (mergeEditor)
            mergeEditors(editorsUUID);
        //
        if (clickedLeftCornerResizeable) {
            glfwSetCursor(window, m_cursors.crossHair);
        } else if (hoveredLeftCornerResizeable) {
            glfwSetCursor(window, m_cursors.crossHair);
        } else if (anyResizeHovered) {
            glfwSetCursor(window, horizontalResizeHovered ? m_cursors.resizeHorizontal : m_cursors.resizeVertical);
        } else {
            glfwSetCursor(window, m_cursors.arrow);
        }
    }


    void Application::resizeEditors(bool anyCornerClicked) {
        bool isValidResizeAll = true;
        for (auto &editor: m_editors) {
            if (editor->ui()->resizeActive && (!anyCornerClicked || editor->ui()->splitting)) {
                auto createInfo = getNewEditorCreateInfo(editor);
                if (!Editor::isValidResize(createInfo, editor))
                    isValidResizeAll = false;
            }
        }
        if (isValidResizeAll) {
            for (auto &editor: m_editors) {
                if (editor->ui()->resizeActive && (!anyCornerClicked || editor->ui()->splitting)) {
                    auto createInfo = getNewEditorCreateInfo(editor);
                    editor->resize(createInfo);
                }
            }
        }
    }

    void Application::splitEditor(uint32_t splitEditorIndex) {
        auto &editor = m_editors[splitEditorIndex];
        EditorCreateInfo &editorCreateInfo = editor->getCreateInfo();
        EditorCreateInfo newEditorCreateInfo(m_guiResources, this, &m_sharedContextData, m_vulkanDevice, &m_allocator,
                                             m_frameBuffers.data());

        EditorCreateInfo::copy(&newEditorCreateInfo, &editorCreateInfo);

        if (editor->ui()->dragHorizontal) {
            editorCreateInfo.width -= editor->ui()->dragDelta.x;
            editorCreateInfo.x += editor->ui()->dragDelta.x;
            newEditorCreateInfo.width = editor->ui()->dragDelta.x;
        } else {
            editorCreateInfo.height += editor->ui()->dragDelta.y;
            newEditorCreateInfo.height = -editor->ui()->dragDelta.y;
            newEditorCreateInfo.y = editorCreateInfo.height + editorCreateInfo.y;
        }
        newEditorCreateInfo.editorIndex = m_editors.size();
        if (!editor->validateEditorSize(newEditorCreateInfo) ||
            !editor->validateEditorSize(editorCreateInfo))
            return;
        editor->resize(editorCreateInfo);
        auto newEditor = createEditor(newEditorCreateInfo);
        editor->ui()->resizeActive = true;
        newEditor->ui()->resizeActive = true;
        editor->ui()->active = true;
        newEditor->ui()->active = true;
        editor->ui()->splitting = true;
        newEditor->ui()->splitting = true;


        if (editor->ui()->dragHorizontal) {
            editor->ui()->lastClickedBorderType = EditorBorderState::Left;
            newEditor->ui()->lastClickedBorderType = EditorBorderState::Right;
        } else {
            editor->ui()->lastClickedBorderType = EditorBorderState::Bottom;
            newEditor->ui()->lastClickedBorderType = EditorBorderState::Top;
        }

        // Copy UI states

        newEditor->onSceneLoad(std::shared_ptr<Scene>(m_activeScene));
        m_editors.push_back(std::move(newEditor));
    }

    void Application::mergeEditors(const std::array<UUID, 2> &mergeEditorIndices) {
        UUID id1 = mergeEditorIndices[0];
        UUID id2 = mergeEditorIndices[1];
        auto &editor1 = findEditorByUUID(id1);
        auto &editor2 = findEditorByUUID(id2);
        if (!editor1 || !editor2) {
            Log::Logger::getInstance()->info("Wanted to merge editors: {} and {} but they were not found",
                                             id1.operator std::string(), id2.operator std::string());
            return;
        }
        editor1->ui()->shouldMerge = false;
        editor2->ui()->shouldMerge = false;
        auto &ci1 = editor1->getCreateInfo();
        auto &ci2 = editor2->getCreateInfo();
        int32_t newX = std::min(ci1.x, ci2.x);
        int32_t newY = std::min(ci1.y, ci2.y);
        int32_t newWidth = ci1.width + ci2.width;
        int32_t newHeight = ci1.height + ci2.height;

        if (editor1->ui()->lastRightClickedBorderType & EditorBorderState::HorizontalBorders) {
            ci1.height = newHeight;
        } else if (editor1->ui()->lastRightClickedBorderType & EditorBorderState::VerticalBorders) {
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

    EditorCreateInfo Application::getNewEditorCreateInfo(std::unique_ptr<Editor> &editor) {
        EditorCreateInfo newEditorCI(m_guiResources, this, &m_sharedContextData, m_vulkanDevice, &m_allocator,
                                     m_frameBuffers.data());
        EditorCreateInfo::copy(&newEditorCI, &editor->getCreateInfo());

        switch (editor->ui()->lastClickedBorderType) {
            case EditorBorderState::Left:
                newEditorCI.x = editor->ui()->x + editor->ui()->cursorDelta.x;
                newEditorCI.width = editor->ui()->width - editor->ui()->cursorDelta.x;
                break;
            case EditorBorderState::Right:
                newEditorCI.width = editor->ui()->width + editor->ui()->cursorDelta.x;
                break;
            case EditorBorderState::Top:
                newEditorCI.y = editor->ui()->y + editor->ui()->cursorDelta.y;
                newEditorCI.height = editor->ui()->height - editor->ui()->cursorDelta.y;
                break;
            case EditorBorderState::Bottom:
                newEditorCI.height = editor->ui()->height + editor->ui()->cursorDelta.y;
                break;
            default:
                Log::Logger::getInstance()->trace(
                        "Resize is somehow active but we have not clicked any borders: {}",
                        editor->getCreateInfo().editorIndex);
                break;
        }
        return newEditorCI;
    }

    void Application::mouseMoved(float x, float y, bool &handled) {
        mouse.insideApp = !(x < 0 || x > m_width || y < 0 || y > m_height);
        float dx = x - mouse.x;
        float dy = y - mouse.y;
        mouse.dx += dx;
        mouse.dy += dy;
        Log::Logger::getInstance()->trace("Cursor velocity: ({},{}), pos: ({},{})", mouse.dx, mouse.dy, mouse.x,
                                          mouse.y);

        for (auto &editor: m_editors) {
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

    void Application::mouseScroll(float change) {
        ImGuiIO &io = ImGui::GetIO();
        io.MouseWheel += 0.5f * static_cast<float>(change);
        for (auto &editor: m_editors) {
            editor->onMouseScroll(change);
        }
        /*
        if (m_guiManager->handles.renderer3D) {
            m_cameras[m_selectedCameraTag].setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
        }
         */
    }


    std::unique_ptr<Editor> &Application::findEditorByUUID(const UUID &uuid) {
        for (auto &editor: m_editors) {
            if (uuid == editor->getUUID()) {
                return editor;
            }
        } // TODO what to return?
    }


    void Application::postRenderActions() {
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

    void Application::keyboardCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
        for (auto &editor: m_editors) {
            ImGui::SetCurrentContext(editor->guiContext());
            ImGuiIO &io = ImGui::GetIO();
            io.AddKeyEvent(ImGuiKey_ModShift, (mods & GLFW_MOD_SHIFT) != 0);
            io.AddKeyEvent(ImGuiKey_ModAlt, (mods & GLFW_MOD_ALT) != 0);
            io.AddKeyEvent(ImGuiKey_ModSuper, (mods & GLFW_MOD_SUPER) != 0);
            io.AddKeyEvent(ImGuiKey_LeftCtrl, (mods & GLFW_MOD_CONTROL) != 0);
            key = ImGui_ImplGlfw_TranslateUntranslatedKey(key, scancode);
            ImGuiKey imgui_key = ImGui_ImplGlfw_KeyToImGuiKey(key);
            io.AddKeyEvent(imgui_key, (action == GLFW_PRESS) || (action == GLFW_REPEAT));
            editor->onKeyCallback(input);
        }
        ImGui::SetCurrentContext(m_mainEditor->guiContext());
        ImGuiIO &io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiKey_ModShift, (mods & GLFW_MOD_SHIFT) != 0);
        io.AddKeyEvent(ImGuiKey_ModAlt, (mods & GLFW_MOD_ALT) != 0);
        io.AddKeyEvent(ImGuiKey_ModSuper, (mods & GLFW_MOD_SUPER) != 0);
        io.AddKeyEvent(ImGuiKey_LeftCtrl, (mods & GLFW_MOD_CONTROL) != 0);
        key = ImGui_ImplGlfw_TranslateUntranslatedKey(key, scancode);
        ImGuiKey imgui_key = ImGui_ImplGlfw_KeyToImGuiKey(key);
        io.AddKeyEvent(imgui_key, (action == GLFW_PRESS) || (action == GLFW_REPEAT));
    }

    std::shared_ptr<Scene> Application::activeScene() {
        return m_activeScene;
    }

    std::shared_ptr<Scene> Application::newScene() {
        if (m_activeScene) {
            m_activeScene.reset();
        }
        m_activeScene = std::make_shared<Scene>("New Scene", this);
        for (auto& editors : m_editors) {
            editors->onSceneLoad(m_activeScene); // Also resets editor camera etc..
            editors->ui()->shared->m_selectedEntity = Entity(); // Reset seleciton context
        }

        return m_activeScene;
    }

    void Application::onFileDrop(const std::filesystem::path &path) {
        for (auto &editor: m_editors) {
            Editor::handleHoverState(editor, mouse);
        }

        for (auto &editor: m_editors) {
            if (editor->ui()->hovered) {
                // Add drop event here
                editor->onFileDrop(path);
            }
        }
    }

    void Application::onCharInput(unsigned int codepoint) {
        for (auto &editor: m_editors) {
            ImGui::SetCurrentContext(editor->guiContext());
            ImGuiIO &io = ImGui::GetIO();
            io.AddInputCharacter(static_cast<unsigned short>(codepoint));
        }
        ImGui::SetCurrentContext(m_mainEditor->guiContext());
        ImGuiIO &io = ImGui::GetIO();
        io.AddInputCharacter(static_cast<unsigned short>(codepoint));
    }


    bool Application::isCurrentProject(std::string projectName) {
        return false;
    }
};
