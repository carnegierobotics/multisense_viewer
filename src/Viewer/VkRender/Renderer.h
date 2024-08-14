/**
 * @file: MultiSense-Viewer/include/Viewer/VkRender/Renderer.h
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

#include "Viewer/VkRender/pch.h"

#include <GLFW/glfw3.h>
#include <entt/entt.hpp>


#include "Viewer/VkRender/Core/VulkanRenderer.h"
#include "Viewer/Scenes/ScriptSupport/ScriptBuilder.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/VkRender/UsageMonitor.h"
#include "Viewer/VkRender/Scene.h"
#include "Viewer/VkRender/Core/RendererConfig.h"
#include "Viewer/VkRender/Core/Camera.h"
#include "Viewer/VkRender/Core/UUID.h"
#include "Viewer/VkRender/Editor.h"
#include "Viewer/VkRender/Editors/EditorFactory.h"

namespace VkRender {
    class Entity;


    class Renderer : VulkanRenderer {

    public:

        /**
         * @brief Default constructor for renderer
         * @param title Title of application
         */
        explicit Renderer(const std::string &title);

        ~Renderer() override = default;

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

        void closeApplication() override {
            VulkanRenderer::closeApplication();
        }

        std::unique_ptr<Editor> createEditor(EditorCreateInfo &createInfo);

        std::unique_ptr<Editor> createEditorWithUUID(UUID uuid, EditorCreateInfo &createInfo);

        std::vector<SwapChainBuffer> &swapChainBuffers() { return swapchain->buffers; }

        VmaAllocator &allocator() { return m_allocator; }

        VulkanDevice &vkDevice() { return *m_vulkanDevice; }

        ImGuiContext *getMainUIContext() { return m_mainEditor->guiContext(); }
        uint32_t currentFrameIndex(){return currentFrame;}

        /*

        Camera &createNewCamera(const std::string &name, uint32_t width, uint32_t height);

        Camera &getCamera();

        Camera &getCamera(std::string tag);
        */

        std::shared_ptr<UsageMonitor> m_usageMonitor; // TODO make private, used widely in imgui code to record user actions

        void loadScene(std::filesystem::path string);

        std::shared_ptr<Scene> activeScene();

        void deleteScene(std::filesystem::path scenePath);

    private:


        void keyboardCallback(GLFWwindow *window, int key, int scancode, int action, int mods) override;

        void onRender() override;

        void updateUniformBuffers() override;

        void windowResized(int32_t dx, int32_t dy, double widthScale, double heightScale) override;

        void addDeviceFeatures() override;

        void mouseMoved(float x, float y, bool &handled) override;

        void mouseScroll(float change) override;

        void postRenderActions() override;

    private:
        //std::unordered_map<std::string, Camera> m_cameras;
        //std::unique_ptr<VkRender::GuiManager> m_guiManager{};
        std::vector<std::unique_ptr<Editor>> m_editors;
        std::vector<std::shared_ptr<Scene>> m_scenes;
        std::string m_selectedCameraTag = "Default Camera";
        std::unique_ptr<EditorFactory> m_editorFactory;

        std::unique_ptr<Editor> m_mainEditor;

        std::shared_ptr<GuiResources> m_guiResources;
        SharedContextData m_sharedContextData;

        friend class RendererConfig;

        void updateEditors();

        EditorCreateInfo getNewEditorCreateInfo( std::unique_ptr<Editor> &editor);

        void resizeEditors(bool anyCornerClicked);

        void splitEditor(uint32_t splitEditorIndex);

        void handleEditorResize();

        void recreateEditor(std::unique_ptr<Editor> &editor, EditorCreateInfo &createInfo);

        void loadEditorSettings(const std::filesystem::path &filePath);

        void mergeEditors(const std::array<UUID, 2> &mergeEditorIndices);

        std::unique_ptr<Editor>& findEditorByUUID(const UUID &uuid);

    };
}


#endif // MultiSense_Viewer_RENDERER_H
