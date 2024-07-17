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

        void closeApplication() override{
            VulkanRenderer::closeApplication();
        }

        Entity createEntity(const std::string &name);
        void destroyEntity(Entity entity);
        Entity createEntityWithUUID(UUID uuid, const std::string &name);
        VkRender::Entity findEntityByName(std::string_view name);
        void markEntityForDestruction(Entity entity);
        void addUILayer(const std::string &layerName);

        entt::registry& registry() {return m_registry;}
        RenderUtils& data() {return m_renderUtils;}
        std::vector<SwapChainBuffer>& swapChainBuffers(){return swapchain->buffers;}
        VmaAllocator& allocator(){return m_allocator;}
        VkDevice &vkDevice() {return device;}

        //GuiObjectHandles& UIContext() {return m_guiManager->handles;}

        Camera &createNewCamera(const std::string &name, uint32_t width, uint32_t height);
        Camera& getCamera();
        Camera& getCamera(std::string tag);

        std::shared_ptr<UsageMonitor> m_usageMonitor; // TODO make private, used widely in imgui code to record user actions

    private:
        template<typename T>
        void onComponentAdded(Entity entity, T &component);

        void keyboardCallback(GLFWwindow *window, int key, int scancode, int action, int mods) override;

        void recordCommands() override;

        bool compute() override;

        void updateUniformBuffers() override;

        void windowResized() override;

        void addDeviceFeatures() override;

        void buildCommandBuffers() override;

        void mouseMoved(float x, float y, bool &handled) override;

        void mouseScroll(float change) override;

        void postRenderActions() override;
        void processDeletions();

        template<typename T>
        bool tryCleanupAndDestroy(Entity &entity, int currentFrame);

    private:
        std::unordered_map<std::string, Camera> m_cameras;
        //std::unique_ptr<VkRender::GuiManager> m_guiManager{};

        std::vector<std::unique_ptr<Scene>> m_scenes;
        std::vector<std::unique_ptr<Editor>> m_editors;
        entt::registry m_registry;
        std::unordered_map<UUID, entt::entity> m_entityMap;
        std::string m_selectedCameraTag = "Default Camera";
        RenderUtils m_renderUtils{};

        std::unique_ptr<Editor> m_mainEditor;

        std::shared_ptr<GuiResources> m_guiResources;

        friend class Entity;


        void createColorResources();

        void setupDepthStencil();

        void handleViewportResize(float x, float y);

    };
}


#endif // MultiSense_Viewer_RENDERER_H
