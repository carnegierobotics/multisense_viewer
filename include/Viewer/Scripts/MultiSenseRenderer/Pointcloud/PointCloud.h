/**
 * @file: MultiSense-Viewer/include/Viewer/Scripts/MultiSenseRenderer/Pointcloud/PointCloud.h
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
 *   2022-11-4, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_POINTCLOUD_H
#define MULTISENSE_POINTCLOUD_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/CRLCameraModels.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/CRLCamera/CRLPhysicalCamera.h"

class PointCloud: public VkRender::Base, public VkRender::RegisteredInFactory<PointCloud>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    PointCloud() {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
        s_bRegistered;
        DISABLE_WARNING_POP    }

    void onDestroy() override{

    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<PointCloud>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "PointCloud"; }
    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with the renderer **/
    VkRender::ScriptTypeFlags getType() override { return type; }
    VkRender::CRL_SCRIPT_DRAW_METHOD getDrawMethod() override {return drawMethod;}    /** @brief UI update function called once per frame **/
    void onUIUpdate(VkRender::GuiObjectHandles *uiHandle) override;
    /** @brief Method to enable/disable drawing of this script **/
    /** @brief set if this script should be drawn or not. */
    void setDrawMethod(VkRender::CRL_SCRIPT_DRAW_METHOD _drawMethod) override{ this->drawMethod = _drawMethod; }

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    VkRender::ScriptTypeFlags type = VkRender::CRL_SCRIPT_TYPE_DEFAULT;
    VkRender::CRL_SCRIPT_DRAW_METHOD drawMethod = VkRender::CRL_SCRIPT_DONT_DRAW;

    std::unique_ptr<CRLCameraModels::Model> model;

    int16_t remoteHeadIndex = 0;
    std::vector<std::string> startedSources{};
    VkRender::Page selectedPreviewTab = VkRender::CRL_TAB_NONE;
    VkRender::CRLCameraResolution res = VkRender::CRL_RESOLUTION_NONE;
    std::future<bool> prepareTexFuture;

    int lumaOrColor = false; // 0 : luma // 1 : Color
    float pointSize = 1.8f;
    bool flipPointCloud = false;
    void draw(CommandBuffer * commandBuffer, uint32_t i, bool b) override;
    bool updatePointCloudParameters(VkRender::Device& dev);

    int point = 0;

    void prepareTexture(VkRender::Device &dev);
};


#endif //MULTISENSE_POINTCLOUD_H
