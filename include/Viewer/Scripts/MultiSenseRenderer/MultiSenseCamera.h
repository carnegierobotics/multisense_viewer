/**
 * @file: MultiSense-Viewer/include/Viewer/Scripts/MultiSenseRenderer/MultiSenseCamera.h
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
 *   2022-09-12, mgjerde@carnegierobotics.com, Created file.
 **/
#pragma once

#include <future>

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"

class MultiSenseCamera: public VkRender::Base, public VkRender::RegisteredInFactory<MultiSenseCamera>, GLTFModel
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    MultiSenseCamera() {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
        s_bRegistered;
        DISABLE_WARNING_POP
    }
    ~MultiSenseCamera()= default;

    void onDestroy() override{
        /*
        cancelLoadModels = true;
        // Wait for async models to finish loading before destorying script.
        // So we dont rush cleaning up vulkan resources for old window before this script finished loading
        while(loadModelFuture.valid() && loadModelFuture.wait_for(std::chrono::duration<float>(0)) != std::future_status::ready);
        S27.reset();
        S30.reset();
        KS21.reset();
        delete deviceCopy;
        */
    }

    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<MultiSenseCamera>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "MultiSenseCamera"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with the renderer **/
    VkRender::ScriptTypeFlags getType() override { return type; }
    VkRender::CRL_SCRIPT_DRAW_METHOD getDrawMethod() override {return drawMethod;}
    void setDrawMethod(VkRender::CRL_SCRIPT_DRAW_METHOD _drawMethod) override{ this->drawMethod = _drawMethod; }

    void onWindowResize(const VkRender::GuiObjectHandles *uiHandle) override;

    void onUIUpdate(VkRender::GuiObjectHandles *uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    VkRender::ScriptTypeFlags type = VkRender::CRL_SCRIPT_TYPE_DISABLED;
    VkRender::CRL_SCRIPT_DRAW_METHOD drawMethod = VkRender::CRL_SCRIPT_DONT_DRAW;
    std::unique_ptr<GLTFModel::Model> S27;
    std::unique_ptr<GLTFModel::Model> S30;
    std::unique_ptr<GLTFModel::Model> KS21;

    VulkanDevice* deviceCopy;
    std::string selectedModel = "Multisense-KS21";
    int selection = 0;
    float alpha = 0.94f;
    int rateTableIndex = -1;
    bool sampleRateChanged = false;
    char sampleRateLabel[25];
    std::vector<uint32_t> rates;

    struct LightSource {
        glm::vec3 color = glm::vec3(1.0f);
        glm::vec3 rotation = glm::vec3(75.0f, 40.0f, 0.0f);
    } lightSource;
    bool imuEnabled = false;
    VkRender::Page selectedPreviewTab = VkRender::CRL_TAB_NONE;
    VkRender::IMUData rot{};
    bool stopDraw = false;
    float frameRate = 30.0f;
    glm::mat4 model;

    std::future<bool> imuRotationFuture;
    std::future<void> setImuConfigFuture;
    std::atomic<bool> cancelLoadModels{false};

    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> calcImuRotationTimer;


    void draw(CommandBuffer * commandBuffer, uint32_t i, bool b) override;

    void loadModelsAsync();
    std::future<void> loadModelFuture;

    void handleIMUUpdate();

    void setIMUSampleRate();
};
