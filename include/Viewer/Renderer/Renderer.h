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
#include <iostream>
#include <thread>
#include <fstream>
#include <filesystem>

#ifdef APIENTRY
#undef APIENTRY
#endif

#include <GLFW/glfw3.h>
#include <entt/entt.hpp>
#include <complex>

#include "Viewer/Core/VulkanRenderer.h"
#include "Viewer/Scripts/Private/Base.h"
#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/CRLCamera/CameraConnection.h"
#include "Viewer/Renderer/UsageMonitor.h"
#include "Viewer/Core/RendererConfig.h"

#include "Generated/ScriptHeader.h"
#include "Viewer/Core/UUID.h"

namespace VkRender {
    class Entity;


    class Renderer : VulkanRenderer {

    public:

        /**
         * @brief Default constructor for renderer
         * @param title Title of application
         */
        explicit Renderer(const std::string &title);

        ~Renderer() = default;

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

        Entity createEntity(const std::string &name);
        void destroyEntity(Entity entity);
        Entity createEntityWithUUID(UUID uuid, const std::string &name);

    public:
        entt::registry m_registry;
        std::unordered_map<UUID, entt::entity> m_entityMap;

    private:
        template<typename T>
        void onComponentAdded(Entity entity, T &component);


        void recordCommands() override;

        bool compute() override;

        void updateUniformBuffers() override;

        void prepareRenderer();


        void windowResized() override;

        void addDeviceFeatures() override;

        void buildCommandBuffers() override;

        void mouseMoved(float x, float y, bool &handled) override;

        void mouseScroll(float change) override;

        void createSelectionFramebuffer();

        void createSelectionImages();

        void destroySelectionBuffer();

        void createSelectionBuffer();

        /**
         * @brief creates instances from classes located in src/Scripts/ directory.
         * Usually each class here represents object(s) in the scene
         */
        void buildScript(const std::string &scriptName);

        /**
         * @brief deletes a script if stored in \refitem builtScriptNames
         * @param scriptName m_Name of script to delete
         */
        void deleteScript(const std::string &scriptName);

        static void
        setScriptDrawMethods(const std::map<std::string, VkRender::CRL_SCRIPT_DRAW_METHOD> &scriptDrawSettings,
                             std::map<std::string, std::shared_ptr<VkRender::Base>> &scripts);

    private:

        std::unique_ptr<VkRender::GuiManager> guiManager{};
        std::map<std::string, std::shared_ptr<VkRender::Base>> scripts{};
        std::map<std::string, std::shared_ptr<VkRender::Base>> scriptsForDeletion{};
        std::vector<std::string> builtScriptNames;
        std::vector<std::string> availableScriptNames;
        std::shared_ptr<UsageMonitor> usageMonitor;
        std::unique_ptr<VkRender::MultiSense::CameraConnection> cameraConnection{};
        VkRender::RenderData renderData{};
        VkRender::RenderUtils renderUtils{};
        VkRender::TopLevelScriptData topLevelScriptData{};

        bool renderSelectionPass = true;

        // Create a host-visible staging buffer that contains the raw m_Image data
        VkBuffer selectionBuffer{};
        VkDeviceMemory selectionMemory{};
        VkBufferImageCopy bufferCopyRegion{};
        VkMemoryRequirements m_MemReqs{};

        friend class Entity;



    };
}


#endif // MultiSense_Viewer_RENDERER_H
