/**
 * @file: MultiSense-Viewer/src/Scripts/Objects/MultiSenseCamera.cpp
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

#include "Viewer/Scripts/Objects/MultiSenseCamera.h"
#include "Viewer/ImGui/ScriptUIAddons.h"

void MultiSenseCamera::setup() {

    loadModelFuture = std::async(std::launch::async,  &MultiSenseCamera::loadModelsAsync, this);
    /*
    Widgets::make()->text("Select other camera models");
    Widgets::make()->slider("##Select model", &selection, 0, 2);
    */
    deviceCopy = new VulkanDevice(renderUtils.device);

}

void MultiSenseCamera::loadModelsAsync(){

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("Scene/spv/object.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("Scene/spv/object.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};

    S30 = std::make_unique<GLTFModel::Model>(&renderUtils, deviceCopy);
    S30->loadFromFile(Utils::getAssetsPath().append("Models/s30_pbr.gltf").string(),deviceCopy,
                     deviceCopy->m_TransferQueue, 1.0f);
    S30->createRenderPipeline(renderUtils, shaders);

    S27 = std::make_unique<GLTFModel::Model>(&renderUtils, deviceCopy);
    S27->loadFromFile(Utils::getAssetsPath().append("Models/s27_pbr.gltf").string(), deviceCopy,
                     deviceCopy->m_TransferQueue, 1.0f);
    S27->createRenderPipeline(renderUtils, shaders);

    KS21 = std::make_unique<GLTFModel::Model>(&renderUtils, deviceCopy);
    KS21->loadFromFile(Utils::getAssetsPath().append("Models/ks21_pbr.gltf").string(), deviceCopy,
                      deviceCopy->m_TransferQueue, 1.0f);
    KS21->createRenderPipeline(renderUtils, shaders);

}

void MultiSenseCamera::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (loadModelFuture.valid() && loadModelFuture.wait_for(std::chrono::duration<float>(0)) != std::future_status::ready)
        return;
    if (selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD && b && !stopDraw){
        if (selectedModel == "Multisense-S30")
            S30->draw(commandBuffer, i);
        else if(selectedModel == "Multisense-S27")
            S27->draw(commandBuffer, i);
        else if(selectedModel == "Multisense-KS21")
            KS21->draw(commandBuffer, i);
        /*
        if (selection == 0)
            S30->draw(commandBuffer, i);
        else if(selection == 1)
            S27->draw(commandBuffer, i);
        else if(selection == 2)
            KS21->draw(commandBuffer, i);
        */
         else{
            Log::Logger::getInstance()->warning("No 3D model corresponding to {}. Not drawing anything", selectedModel);
            stopDraw = true;
        }
    }
}

void MultiSenseCamera::update() {
    auto &d = bufferOneData;
    d->model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
    d->model = glm::rotate(d->model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    //d->model = glm::rotate(d->model, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    d->model = glm::scale(d->model, glm::vec3(0.001f, 0.001f, 0.001f));

    if (imuEnabled) {
        renderData.crlCamera->getImuRotation(&rot, 0);
        //printf("Pitch, Roll:  (%f, %f): Orig: (%f, %f)\n", static_cast<double>(P), static_cast<double>(R), static_cast<double>(rot.pitch), static_cast<double>(rot.roll));
        d->model = glm::rotate(d->model, static_cast<float>(-rot.roll), glm::vec3(1.0f, 0.0f, 0.0f));
        d->model = glm::rotate(d->model, static_cast<float>(rot.pitch), glm::vec3(0.0f, 1.0f, 0.0f));
    } else {
        //d->model = glm::rotate(d->model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    }
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;
    d->camPos = glm::vec3(
            static_cast<double>(-renderData.camera->m_Position.z) * sin(
                    static_cast<double>(glm::radians(renderData.camera->m_Rotation.y))) *
            cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.x))),
            static_cast<double>(-renderData.camera->m_Position.z) * sin(
                    static_cast<double>(glm::radians(renderData.camera->m_Rotation.x))),
            static_cast<double>(renderData.camera->m_Position.z) *
            cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.y))) *
            cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.x)))
    );

    auto &d2 = bufferTwoData;
    d2->lightDir = glm::vec4(
            static_cast<double>(sinf(glm::radians(lightSource.rotation.x))) * cos(
                    static_cast<double>(glm::radians(lightSource.rotation.y))),
            sin(static_cast<double>(glm::radians(lightSource.rotation.y))),
            cos(static_cast<double>(glm::radians(lightSource.rotation.x))) * cos(
                    static_cast<double>(glm::radians(lightSource.rotation.y))),
            0.0f);


    auto *ptr = reinterpret_cast<VkRender::FragShaderParams *>(sharedData->data);
    d2->gamma = ptr->gamma;
    d2->exposure = ptr->exposure;
    d2->scaleIBLAmbient = ptr->scaleIBLAmbient;
    d2->debugViewInputs = ptr->debugViewInputs;
    d2->prefilteredCubeMipLevels = renderUtils.skybox.prefilteredCubeMipLevels;


}


void MultiSenseCamera::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    bool noActiveDevice = true;
    for (const auto &d: uiHandle->devices) {
        if (d.state != CRL_STATE_ACTIVE)
            continue;
        selectedPreviewTab = d.selectedPreviewTab;
        selectedModel = d.cameraName;
        imuEnabled = d.useIMU;
        noActiveDevice = false;
        stopDraw = false;
    }
    if (noActiveDevice)
        stopDraw = true;
}