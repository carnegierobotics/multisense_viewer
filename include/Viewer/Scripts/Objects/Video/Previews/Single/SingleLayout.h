/**
 * @file: MultiSense-Viewer/include/Viewer/Scripts/Objects/Video/Previews/Single/SingleLayout.h
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
 *   2022-09-16, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_VIEWER_SINGLELAYOUT_H
#define MULTISENSE_VIEWER_SINGLELAYOUT_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/CRLCameraModels.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/CRLCamera/CRLPhysicalCamera.h"

class SingleLayout: public VkRender::Base, public VkRender::RegisteredInFactory<SingleLayout>, CRLCameraModels
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    SingleLayout() {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
        s_bRegistered;
        DISABLE_WARNING_POP}
    void onDestroy() override{
        stbi_image_free(m_NoDataTex);
        stbi_image_free(m_NoSourceTex);

    }
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<SingleLayout>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "SingleLayout"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}
    /** @brief called after renderer has handled a window resize event **/
    void onWindowResize(const VkRender::GuiObjectHandles *uiHandle) override;
    /** @brief Method to enable/disable drawing of this script **/
    void setDrawMethod(ScriptType _type) override{ this->type = _type; }

    void onUIUpdate(VkRender::GuiObjectHandles *uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = CRL_SCRIPT_TYPE_DEFAULT;

    std::unique_ptr<CRLCameraModels::Model> m_Model;
    std::unique_ptr<CRLCameraModels::Model> m_ModelTestDevice; // For use with test device i.e. not a MultiSense camera debug version
    std::unique_ptr<CRLCameraModels::Model> m_NoDataModel;
    std::unique_ptr<CRLCameraModels::Model> m_NoSourceModel;
    enum {
        DRAW_NO_SOURCE = 0,
        DRAW_NO_DATA = 1,
        DRAW_MULTISENSE = 2,
        DRAW_TEST = 3
    } state;

    float up = -1.3f;
    unsigned char* m_NoDataTex{};
    unsigned char* m_TestDeviceTex{};
    unsigned char* m_NoSourceTex{};

    Page selectedPreviewTab = CRL_TAB_NONE;
    float posY = 0.0f;
    float scaleX = 0.25f;
    float scaleY = 0.25f;
    float centerX = 0.0f;
    float centerY = 0.0f;
    std::string src;
    int16_t remoteHeadIndex = 0;
    CRLCameraResolution res = CRL_RESOLUTION_NONE;
    int texWidth = 0, texHeight = 0, texChannels = 0;
    CRLCameraDataType textureType = CRL_CAMERA_IMAGE_NONE;
    int64_t lastPresentedFrameID = -1;
    std::chrono::steady_clock::time_point lastPresentTime;

    bool virtualDevice = false;
    bool zoomEnabled = false;

    VkRender::ZoomParameters zoom{};
    const VkRender::ImageEffectOptions* options{};

    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;

    /** @brief Updates PosX-Y variables to match the desired positions before creating the quad. Using positions from ImGui */
    void transformToUISpace(const VkRender::GuiObjectHandles * handles, const VkRender::Device& element);

    void prepareMultiSenseTexture();

    void prepareDefaultTexture();

    void updateLog() const;

    void prepareTestDeviceTexture();
};



#endif //MULTISENSE_VIEWER_SINGLELAYOUT_H
