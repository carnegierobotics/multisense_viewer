/**
 * @file: MultiSense-Viewer/include/Viewer/Scripts/Private/Base.h
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
 *   2021-11-21, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_BASE_H
#define MULTISENSE_BASE_H

#include <filesystem>
#include <utility>
#include <stb_image.h>
#include <stb_image_write.h>
#ifdef APIENTRY
#undef APIENTRY
#endif
#include <GLFW/glfw3.h>

#include "Viewer/Core/KeyInput.h"
#include "Viewer/Core/Camera.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/CRLCamera/CRLPhysicalCamera.h"
#include "Viewer/Scripts/Private/SharedData.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/ImGui/Layer.h"

#define TOLERATE_FRAME_NUM_SKIP 10 // 10 frames means 2.5 for remote head. Should probably bet set based on remote head or not
#define SHARED_MEMORY_SIZE_1MB 1000000

namespace VkRender {
    /**
     * @brief Base class for scripts that can be attached to renderer. See @refitem Example for how to implement a script.
     */
    class Base {
    public:
        std::vector<ScriptBufferSet> ubo;

        std::vector<VkShaderModule> shaderModules{};
        VkRender::SkyboxTextures skyboxTextures;

        VkRender::RenderUtils renderUtils{};
        VkRender::RenderData renderData{};
        std::unique_ptr<SharedData> sharedData; // TODO remove this
        VkRender::TopLevelScriptData* topLevelData;

        virtual ~Base() = default;


        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
        DISABLE_WARNING_EMPTY_BODY

        /**@brief Pure virtual function called only once when VK is ready to render*/
        virtual void setup() {
        }

        /**@brief Pure virtual function called once every frame*/
        virtual void update() {
        }

        /**@brief Pure virtual function called each frame*/
        virtual void onUIUpdate(VkRender::GuiObjectHandles* uiHandle) {
        }

        /**@brief Pure virtual function called to enable/disable drawing of this script*/
        virtual void setDrawMethod(VkRender::CRL_SCRIPT_DRAW_METHOD drawMethod) = 0;

        /**@brief Virtual function called when resize event is triggered from the platform os*/
        virtual void onWindowResize(const VkRender::GuiObjectHandles* uiHandle) {
        }

        /**@brief Called once script is requested for deletion */
        virtual void onDestroy() {
        }

        /**@brief Which script type this is. Can be used to flashing/disable rendering of this script */
        virtual ScriptTypeFlags getType() {
            return VkRender::CRL_SCRIPT_TYPE_DISABLED;
        }

        /**@brief Which script type this is. Can be used to flashing/disable rendering of this script */
        virtual VkRender::CRL_SCRIPT_DRAW_METHOD getDrawMethod() {
            return VkRender::CRL_SCRIPT_DONT_DRAW;
        }

        /**@brief Record draw command into a VkCommandBuffer */
        virtual void draw(CommandBuffer* commandBuffer, uint32_t i, bool b) {
            //Log::Logger::getInstance()->info("draw not overridden for {} script", renderData.scriptName);
        }

        DISABLE_WARNING_POP

        void windowResize(VkRender::RenderData* data, const VkRender::GuiObjectHandles* uiHandle) {
            updateRenderData(data);
            onWindowResize(uiHandle);
        }

        void uiUpdate(VkRender::GuiObjectHandles* uiHandle) {
            if (!this->renderData.drawThisScript)
                return;
            if (renderData.crlCamera != nullptr)
                onUIUpdate(uiHandle);
        }

        void drawScript(CommandBuffer* commandBuffer, uint32_t i, bool b) {
            if (!renderData.drawThisScript || getDrawMethod() == VkRender::CRL_SCRIPT_DONT_DRAW)
                return;

            /*
            auto time = std::chrono::steady_clock::now();
            std::chrono::duration<float> time_span =
                    std::chrono::duration_cast<std::chrono::duration<float>>(time - lastLogTime);
            if (time_span.count() > INTERVAL_10_SECONDS || renderData.scriptDrawCount == 0) {
                lastLogTime = std::chrono::steady_clock::now();
                Log::Logger::getInstance()->info("Draw-number: {} for script: {}", renderData.scriptDrawCount, renderData.scriptName.c_str());
            }
             */
            draw(commandBuffer, i, b);
            if (i == 0)
                renderData.scriptDrawCount++;
        }


        void updateUniformBufferData(VkRender::RenderData* data) {
            updateRenderData(data);

            renderData.scriptRuntime = (std::chrono::steady_clock::now() - startTime).count();

            if (renderData.crlCamera != nullptr)
                update();

            VkRender::UniformBufferSet &currentUB = renderUtils.uniformBuffers[renderData.index];
            if (renderData.type != VkRender::CRL_SCRIPT_TYPE_DISABLED) {
                memcpy(currentUB.bufferOne.mapped, ubo[0].mvp.get(), sizeof(VkRender::UBOMatrix));
                memcpy(currentUB.bufferTwo.mapped, ubo[0].fragShader.get(),
                       sizeof(VkRender::FragShaderParams));
                memcpy(currentUB.bufferThree.mapped, ubo[0].pointCloudData.get(),
                       sizeof(VkRender::PointCloudParam));
            }

        }

        void createUniformBuffers(const RenderUtils& utils, RenderData rData, TopLevelScriptData* topLevelPtr) {
            topLevelData = topLevelPtr;
            renderData = std::move(rData);
            renderUtils = utils;
            renderUtils.uniformBuffers.resize(renderUtils.UBCount);
            startTime = std::chrono::steady_clock::now();
            lastLogTime = std::chrono::steady_clock::now();
            int renderPassCount = 2; // TODO make adjustable
            ubo.resize(renderPassCount);
            renderUtils.uboDevice.resize(renderPassCount);
            for (int i = 0; i < renderPassCount; ++i){
                ubo[i].mvp = std::make_unique<VkRender::UBOMatrix>();
                ubo[i].fragShader = std::make_unique<VkRender::FragShaderParams>();
                ubo[i].pointCloudData = std::make_unique<VkRender::PointCloudParam>();

                renderUtils.uboDevice[i].resize(renderUtils.UBCount);

            }


            for (auto& uniformBuffer : renderUtils.uniformBuffers) {
                renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                 &uniformBuffer.bufferOne, sizeof(VkRender::UBOMatrix));
                uniformBuffer.bufferOne.map();

                renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                 &uniformBuffer.bufferTwo, sizeof(VkRender::FragShaderParams));
                uniformBuffer.bufferTwo.map();

                renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                 &uniformBuffer.bufferThree, sizeof(VkRender::PointCloudParam));
                uniformBuffer.bufferThree.map();
            }
            for (int i = 0; i < renderPassCount; ++i) {

                for (auto &uniformBuffer: renderUtils.uboDevice[i]) {
                    renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                     &uniformBuffer.bufferOne, sizeof(VkRender::UBOMatrix));
                    uniformBuffer.bufferOne.map();

                    renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                     &uniformBuffer.bufferTwo, sizeof(VkRender::FragShaderParams));
                    uniformBuffer.bufferTwo.map();

                    renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                     &uniformBuffer.bufferThree, sizeof(VkRender::PointCloudParam));
                    uniformBuffer.bufferThree.map();
                }
            }
            renderData.scriptRuntime = (std::chrono::steady_clock::now() - startTime).count();

            sharedData = std::make_unique<SharedData>(SHARED_MEMORY_SIZE_1MB);

            if (getType() != VkRender::CRL_SCRIPT_TYPE_DISABLED) {
                setup();
                renderData.drawThisScript = true;
            }
        }

        /**@brief Call to delete the attached script. */
        bool onDestroyScript() {
            onDestroy();

            for (auto& shaderModule : shaderModules) {
                vkDestroyShaderModule(renderUtils.device->m_LogicalDevice, shaderModule, nullptr);
            }
            shaderModules.clear();

            for (auto& uniformBuffer : renderUtils.uniformBuffers) {
                uniformBuffer.bufferOne.unmap();
                uniformBuffer.bufferTwo.unmap();
                uniformBuffer.bufferThree.unmap();
            }
            bool allCleanedUp = true;
            if (resourceTracker.empty())
                return true;

            for (size_t i = 0; i < renderUtils.UBCount; ++i) {
                if (!resourceTracker[i].cleanUpReady) {
                    if(resourceTracker[i].cleanUp(renderUtils.device->m_LogicalDevice, (*renderUtils.fence)[i]))
                        Log::Logger::getInstance()->trace("Resources cleaned up for frame: {}", i);

                }

                if (!resourceTracker[i].cleanUpReady) {
                    allCleanedUp = false;
                    Log::Logger::getInstance()->trace("Resources busy in frame: {}, postponing..", i);
                }
            }

            return allCleanedUp;
        }

        /**
         * Utility function to load shaders in scripts. Automatically creates and destroys shaderModule objects if a valid shader file is passed
         * @param fileName
         * @param stage
         * @return
         */
        [[nodiscard]] VkPipelineShaderStageCreateInfo
        loadShader(std::string fileName, VkShaderStageFlagBits stage) {
            // Check if we have .spv extensions. If not then add it.
            std::size_t extension = fileName.find(".spv");
            if (extension == std::string::npos)
                fileName.append(".spv");
            VkShaderModule module;
            Utils::loadShader((Utils::getShadersPath().append(fileName)).string().c_str(),
                              renderUtils.device->m_LogicalDevice, &module);
            assert(module != VK_NULL_HANDLE);

            shaderModules.emplace_back(module);
            VkPipelineShaderStageCreateInfo shaderStage = {};
            shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStage.stage = stage;
            shaderStage.module = module;
            shaderStage.pName = "main";
            Log::Logger::getInstance()->info("Loaded shader {} for stage {}", fileName, static_cast<uint32_t>(stage));
            return shaderStage;
        }


    private:
        struct ResourceEntry {
            // Resource handle (e.g., VkBuffer, VkImage, etc.)
            VkPipeline pipeline = VK_NULL_HANDLE;
            VkPipeline pipeline2 = VK_NULL_HANDLE;
            VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
            VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
            VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;

            bool cleanUpReady = false;

            void destroyResources(const VkDevice& device) const {
                // If not in use then destroy
                if (descriptorSetLayout != VK_NULL_HANDLE)
                    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                if (descriptorPool != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
                if (pipelineLayout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
                if (pipeline != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, pipeline, nullptr);
                if (pipeline2 != VK_NULL_HANDLE)
                    vkDestroyPipeline(device, pipeline2, nullptr);
            }

            bool cleanUp(const VkDevice& device, const VkFence& fence) {
                if (vkGetFenceStatus(device, fence) == VK_SUCCESS) {
                    // The command buffer has finished execution; safe to clean up resources
                    // Perform cleanup operations here
                    destroyResources(device);
                    cleanUpReady = true;
                    return cleanUpReady;
                }
                return false;
            }
        };

        std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> startTime;
        std::chrono::steady_clock::time_point lastLogTime;

        void updateRenderData(VkRender::RenderData* data) { // TODO get rid of this function
            this->renderData.camera = data->camera;
            this->renderData.crlCamera = data->crlCamera;
            this->renderData.deltaT = data->deltaT;
            this->renderData.index = data->index;
            this->renderData.height = data->height;
            this->renderData.width = data->width;
            this->renderData.type = getType();
            this->renderData.renderPassIndex = data->renderPassIndex;
        }

        const Input* input{};

    protected:
        std::vector<ResourceEntry> resourceTracker;
    };
}

#endif //MULTISENSE_BASE_H
