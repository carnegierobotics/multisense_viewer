/**
 * @file: MultiSense-Viewer/include/Viewer/Scripts/Objects/Video/RecordFrames.h
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
 *   2022-10-12, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_VIEWER_RECORDFRAMES_H
#define MULTISENSE_VIEWER_RECORDFRAMES_H


#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/CRLCameraModels.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/CRLCamera/CRLPhysicalCamera.h"
#include "Viewer/Tools/ThreadPool.h"

class RecordFrames : public VkRender::Base, public VkRender::RegisteredInFactory<RecordFrames>, CRLCameraModels {
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    RecordFrames() {
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
        s_bRegistered;
        DISABLE_WARNING_POP
    }

    void onDestroy() override {
    }

    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<RecordFrames>(); }

    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "RecordFrames"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;

    /** @brief update function called once per frame **/
    void update() override;

    /** @brief Get the type of script. This will determine how it interacts with the renderer **/
    ScriptTypeFlags getType() override { return type; }
    DrawMethod getDrawMethod() override {return drawMethod;}

    /** @brief Method to enable/disable drawing of this script **/
    void setDrawMethod(DrawMethod _drawMethod) override { this->drawMethod = _drawMethod; }

    void onUIUpdate(VkRender::GuiObjectHandles *uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptTypeFlags type = CRL_SCRIPT_TYPE_DEFAULT;
    DrawMethod drawMethod = CRL_SCRIPT_DRAW;

    std::unique_ptr<VkRender::ThreadPool> threadPool;

    bool useAuxColor = false; // 0 luma, 1 color
    bool savePointCloud = false;
    bool saveIMUData = false;
    bool saveImage = false;
    bool prevSaveState = false;
    bool isRemoteHead = false;

    std::string saveFolderImage;
    std::filesystem::path saveFolderPointCloud;
    std::filesystem::path saveFolderIMUData;
    std::string compression;
    std::vector<std::string> sources;
    std::vector<std::string> prevSources;
    int prevNumberSources = 0;
    std::unordered_map<std::string, uint32_t> ids;
    crl::multisense::RemoteHeadChannel remoteHeadIndex = 0;
    CRLCameraDataType textureType = CRL_CAMERA_IMAGE_NONE;

    std::vector<std::string> colorSources{"Color Rectified Aux", "Luma Rectified Aux"};
    std::unordered_map<std::string, uint32_t> savedImageSourceCount;
    std::unordered_map<std::string, uint32_t> saveImageCount;
    std::unordered_map<std::string, uint32_t> lastSavedImagesID;


    template<typename T>
    static inline std::array<uint8_t, 3> ycbcrToRGB(uint8_t *luma,
                                             uint8_t *chroma,
                                             const uint32_t &imageWidth,
                                             size_t u,
                                             size_t v) {
        const auto *lumaP = reinterpret_cast<const uint8_t *>(luma);
        const auto *chromaP = reinterpret_cast<const uint8_t *>(chroma);

        const size_t luma_offset = (v * imageWidth) + u;
        const size_t chroma_offset = 2 * (((v / 2) * (imageWidth / 2)) + (u / 2));

        const auto px_y = static_cast<float>(lumaP[luma_offset]);
        const auto px_cb = static_cast<float>(chromaP[chroma_offset + 0]) - 128.0f;
        const auto px_cr = static_cast<float>(chromaP[chroma_offset + 1]) - 128.0f;

        float px_r = px_y + 1.13983f * px_cr;
        float px_g = px_y - 0.39465f * px_cb - 0.58060f * px_cr;
        float px_b = px_y + 2.03211f * px_cb;

        if (px_r < 0.0f) px_r = 0.0f;
        else if (px_r > 255.0f) px_r = 255.0f;
        if (px_g < 0.0f) px_g = 0.0f;
        else if (px_g > 255.0f) px_g = 255.0f;
        if (px_b < 0.0f) px_b = 0.0f;
        else if (px_b > 255.0f) px_b = 255.0f;

        return {{static_cast<uint8_t>(px_r), static_cast<uint8_t>(px_g), static_cast<uint8_t>(px_b)}};
    }

    static inline void
    ycbcrToRGB(uint8_t *luma, uint8_t *chroma, const uint32_t &width,
               const uint32_t &height, uint8_t *output);

    static void
    saveImageToFileAsync(CRLCameraDataType type, const std::string &path, std::string &stringSrc,
                         std::shared_ptr<VkRender::TextureData> &ptr,
                         std::string &compression);

    void
    static savePointCloudToPlyFile(const std::filesystem::path& saveDirectory,
                                   std::shared_ptr<VkRender::TextureData> &depthTex,
                                   std::shared_ptr<VkRender::TextureData> &colorTex,
                                   bool useAuxColor,
                                   const glm::mat4 &Q, const float &scale, const float &focalLength
    );

    size_t hashVector(const std::vector<std::string> &v);

    void saveImageToFile();

    void savePointCloudToFile();

    void saveIMUDataToFile();

    static void saveIMUDataToFileAsync(const std::filesystem::path &saveDirectory,
                                const std::vector<VkRender::MultiSense::CRLPhysicalCamera::ImuData> &gyro,
                                const std::vector<VkRender::MultiSense::CRLPhysicalCamera::ImuData> &accel);
};


#endif //MULTISENSE_VIEWER_RECORDFRAMES_H
