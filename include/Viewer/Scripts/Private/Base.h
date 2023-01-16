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
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include "Viewer/Core/KeyInput.h"
#include "Viewer/CRLCamera/CRLPhysicalCamera.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/Renderer/SharedData.h"
#include "Viewer/Core/Camera.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Logger.h"

#define TOLERATE_FRAME_NUM_SKIP 10 // 10 frames means 2.5 for remote head. Should probably bet set based on remote head or not
#define SHARED_MEMORY_SIZE_1MB 1000000

class CameraConnection; // forward declaration of this class to speed up compile time. Separate Scripts/model_loaders from ImGui source recompile

namespace VkRender {

    /**
     * @brief Base class for scripts that can be attached to renderer. See @refitem Example for how to implement a script.
     */
    class Base {
    public:

        std::unique_ptr<VkRender::UBOMatrix> bufferOneData{};
        std::unique_ptr<VkRender::FragShaderParams> bufferTwoData{};
        std::unique_ptr<VkRender::PointCloudParam> bufferThreeData{};
        std::vector<std::unique_ptr<VkRender::RenderDescriptorBuffers>> additionalBuffers{};
        std::vector<std::vector<VkRender::RenderDescriptorBuffersData>> additionalBuffersData{};

        std::vector<VkShaderModule> shaderModules{};

        VkRender::RenderUtils renderUtils{};
        VkRender::RenderData renderData{};
        std::unique_ptr<SharedData> sharedData;

        virtual ~Base() = default;


        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
        DISABLE_WARNING_EMPTY_BODY

        /**@brief Pure virtual function called only once when VK is ready to render*/
        virtual void setup() {
            if (getType() != CRL_SCRIPT_TYPE_DISABLED);
            //Log::Logger::getInstance()->info("Function setup not overridden for {} script", renderData.scriptName);
        };

        /**@brief Pure virtual function called once every frame*/
        virtual void update() {
            if (getType() != CRL_SCRIPT_TYPE_DISABLED);
            //Log::Logger::getInstance()->info("Function update not overridden for {} script", renderData.scriptName);
        };

        /**@brief Pure virtual function called each frame*/
        virtual void onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
            if (getType() != CRL_SCRIPT_TYPE_DISABLED);
            //Log::Logger::getInstance()->info("Function onUIUpdate not overridden for {} script",                                                 renderData.scriptName);
        };

        /**@brief Pure virtual function called to enable/disable drawing of this script*/
        virtual void setDrawMethod(ScriptType type) {
            if (getType() != CRL_SCRIPT_TYPE_DISABLED);
            //Log::Logger::getInstance()->info("Function setDrawMethod not overridden for {} script",                                                 renderData.scriptName);
        };

        /**@brief Virtual function called when resize event is triggered from the platform os*/
        virtual void onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {
            if (getType() != CRL_SCRIPT_TYPE_DISABLED);
            //Log::Logger::getInstance()->info("Function onWindowResize not overridden for {} script",                                                renderData.scriptName);
        };

        /**@brief Called once script is requested for deletion */
        virtual void onDestroy() {
            if (getType() != CRL_SCRIPT_TYPE_DISABLED);
            //Log::Logger::getInstance()->info("Function onDestroy not overridden for {} script",                                                 renderData.scriptName);

        };

        /**@brief Which script type this is. Can be used to flashing/disable rendering of this script */
        virtual ScriptType getType() {
            return CRL_SCRIPT_TYPE_DISABLED;
        }

        /**@brief Record draw command into a VkCommandBuffer */
        virtual void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
            //Log::Logger::getInstance()->info("draw not overridden for {} script", renderData.scriptName);

        };
        DISABLE_WARNING_POP

        void windowResize(VkRender::RenderData *data, const VkRender::GuiObjectHandles *uiHandle) {
            updateRenderData(data);
            onWindowResize(uiHandle);
        }

        void uiUpdate(const VkRender::GuiObjectHandles *uiHandle) {
            if (!this->renderData.drawThisScript)
                return;
            if (renderData.crlCamera != nullptr)
                onUIUpdate(uiHandle);

        }

        void drawScript(VkCommandBuffer commandBuffer, uint32_t i, bool b) {

            if (!renderData.drawThisScript)
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

        };


        void updateUniformBufferData(VkRender::RenderData *data) {
            updateRenderData(data);

            renderData.scriptRuntime = (std::chrono::steady_clock::now() - startTime).count();

            if (*renderData.crlCamera != nullptr)
                update();
            if (renderData.type == CRL_SCRIPT_TYPE_RENDER)
                update();

            VkRender::UniformBufferSet &currentUB = renderUtils.uniformBuffers[renderData.index];
            if (renderData.type != CRL_SCRIPT_TYPE_DISABLED) {
                memcpy(currentUB.bufferOne.mapped, bufferOneData.get(), sizeof(VkRender::UBOMatrix));
                memcpy(currentUB.bufferTwo.mapped, bufferTwoData.get(), sizeof(VkRender::FragShaderParams));
                memcpy(currentUB.bufferThree.mapped, bufferThreeData.get(), sizeof(VkRender::PointCloudParam));

                // TODO Future optimization could be to copy blocks of data instead of for for loops.
                if (renderData.additionalBuffers) {
                    for (size_t i = 0; i < additionalBuffers.size(); ++i) {
                        memcpy(additionalBuffersData[i][renderData.index].mvp.mapped, &additionalBuffers[i]->mvp,
                               sizeof(VkRender::UBOMatrix));

                        memcpy(additionalBuffersData[i][renderData.index].light.mapped,
                               &additionalBuffers[i]->light,
                               sizeof(VkRender::FragShaderParams));
                    }
                }
            }


        }

        void createUniformBuffers(const VkRender::RenderUtils &utils, VkRender::RenderData rData) {
            renderData = std::move(rData);
            renderUtils = utils;
            renderUtils.uniformBuffers.resize(renderUtils.UBCount);
            startTime = std::chrono::steady_clock::now();
            lastLogTime = std::chrono::steady_clock::now();
            bufferOneData = std::make_unique<VkRender::UBOMatrix>();
            bufferTwoData = std::make_unique<VkRender::FragShaderParams>();
            bufferThreeData = std::make_unique<VkRender::PointCloudParam>();
            for (auto &uniformBuffer: renderUtils.uniformBuffers) {
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
            renderData.scriptRuntime = (std::chrono::steady_clock::now() - startTime).count();

            sharedData = std::make_unique<SharedData>(SHARED_MEMORY_SIZE_1MB);

            if (getType() != CRL_SCRIPT_TYPE_DISABLED) {
                setup();
                renderData.drawThisScript = true;
            }
        }

        /**@brief Call to delete the attached script. */
        void onDestroyScript() {
            for (auto *shaderModule: shaderModules) {
                vkDestroyShaderModule(renderUtils.device->m_LogicalDevice, shaderModule, nullptr);
            }

            for (auto &uniformBuffer: renderUtils.uniformBuffers) {
                uniformBuffer.bufferOne.unmap();
                uniformBuffer.bufferTwo.unmap();
                uniformBuffer.bufferThree.unmap();
            }

            onDestroy();
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
            Utils::loadShader((Utils::getShadersPath() + fileName).c_str(),
                              renderUtils.device->m_LogicalDevice, &module);
            assert(module != VK_NULL_HANDLE);

            shaderModules.emplace_back(module);
            VkPipelineShaderStageCreateInfo shaderStage = {};
            shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStage.stage = stage;
            shaderStage.module = module;
            shaderStage.pName = "main";
            return shaderStage;
        }


        void requestAdditionalBuffers(size_t numBuffers) {
            // Num buffers for each object
            additionalBuffers.resize(numBuffers);
            additionalBuffersData.resize(numBuffers);


            for (size_t i = 0; i < numBuffers; ++i) {
                additionalBuffers[i] = std::make_unique<VkRender::RenderDescriptorBuffers>();
                // Buffers for each swapchain image
                additionalBuffersData[i].resize(renderUtils.uniformBuffers.size());
                for (auto &uniformBuffer: additionalBuffersData[i]) {
                    renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                     &uniformBuffer.mvp, sizeof(VkRender::UBOMatrix));
                    uniformBuffer.mvp.map();

                    renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                     &uniformBuffer.light, sizeof(VkRender::FragShaderParams));
                    uniformBuffer.light.map();
                }
            }

            renderData.additionalBuffers = true;

        }

    private:
        std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> startTime;
        std::chrono::steady_clock::time_point lastLogTime;

        void updateRenderData(VkRender::RenderData *data) {
            this->renderData.camera = data->camera;
            this->renderData.crlCamera = data->crlCamera;
            this->renderData.deltaT = data->deltaT;
            this->renderData.index = data->index;
            this->renderData.height = data->height;
            this->renderData.width = data->width;
            this->renderData.type = getType();
            input = data->input;
        }

        const Input *input{};
    };
};

#endif //MULTISENSE_BASE_H
