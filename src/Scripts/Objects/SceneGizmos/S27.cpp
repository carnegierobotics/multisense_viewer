/**
 * @file: MultiSense-Viewer/src/Scripts/Objects/SceneGizmos/S27.cpp
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

#include "Viewer/Scripts/Objects/SceneGizmos/S27.h"

void S27::setup() {
    m_Skybox = std::make_unique<GLTFModel::Model>(renderUtils.device);
    m_Skybox->loadFromFile(Utils::getAssetsPath() + "Models/Box/glTF-Embedded/Box.gltf", renderUtils.device,
                          renderUtils.device->m_TransferQueue, 1.0f);

    std::vector<VkPipelineShaderStageCreateInfo> envShaders = {{loadShader("Scene/spv/filtercube.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                               {loadShader("Scene/spv/irradiancecube.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)},
                                                               {loadShader("Scene/spv/prefilterenvmap.frag",
                                                                           VK_SHADER_STAGE_FRAGMENT_BIT)},
                                                               {loadShader("Scene/spv/genbrdflut.vert",
                                                                           VK_SHADER_STAGE_VERTEX_BIT)},
                                                               {loadShader("Scene/spv/genbrdflut.frag",
                                                                           VK_SHADER_STAGE_FRAGMENT_BIT)}};

    //m_Skybox->setEnvironmentMap(renderUtils, Utils::getAssetsPath() + "Textures/Environments/papermill.ktx", envShaders);

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("Scene/spv/box.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("Scene/spv/box.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};


    // Obligatory call to prepare render resources for GLTFModel.
    //m_Skybox->createRenderPipeline(renderUtils, shaders);
}

void S27::draw(VkCommandBuffer commandBuffer, uint32_t i, bool primaryDraw) {
    if (primaryDraw)
        ;//m_Skybox->draw(commandBuffer, i);
}

void S27::update() {
    VkRender::UBOMatrix mat{};

    auto &d = bufferOneData;
    d->model = glm::mat4(glm::mat3(renderData.camera->matrices.view));
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;
    auto &d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->m_ViewPos;
    //shaderValuesParams.prefilteredCubeMipLevels = static_cast<float>(numMips);

}


void S27::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const auto &d: uiHandle->devices) {
        if (d.state != CRL_STATE_ACTIVE)
            continue;
    }
}