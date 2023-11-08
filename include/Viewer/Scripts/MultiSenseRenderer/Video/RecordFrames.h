/**
 * @file: MultiSense-Viewer/include/Viewer/Scripts/MultiSenseRenderer/Video/RecordFrames.h
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

DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FUNCTION
#include <RosbagWriter/RosbagWriter.h>
DISABLE_WARNING_POP

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
    std::string fileFormat;
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

    std::shared_ptr<CRLRosWriter::RosbagWriter> writer;
    std::mutex rosbagWriterMutex;


    static void
    saveImageToFileAsync(CRLCameraDataType type, const std::string &path, std::string &stringSrc,
                         std::shared_ptr<VkRender::TextureData> &ptr,
                         std::string &fileFormat, std::shared_ptr<CRLRosWriter::RosbagWriter> rosbagWriter, std::mutex& rosbagWriterMut);

    void
    static savePointCloudToPlyFile(const std::filesystem::path& saveDirectory,
                                   std::shared_ptr<VkRender::TextureData> &depthTex,
                                   std::shared_ptr<VkRender::TextureData> &colorTex,
                                   bool useAuxColor,
                                   const glm::mat4 &Q, const float &scale, const float &focalLength
    );

    static void finishWritingRosbag(std::shared_ptr<CRLRosWriter::RosbagWriter> rosbagWriter, std::mutex& rosbagWriterMut);

    size_t hashVector(const std::vector<std::string> &v);

    void saveImageToFile();

    void savePointCloudToFile();

    void saveIMUDataToFile();

    static void saveIMUDataToFileAsync(const std::filesystem::path &saveDirectory,
                                const std::vector<VkRender::MultiSense::CRLPhysicalCamera::ImuData> &gyro,
                                const std::vector<VkRender::MultiSense::CRLPhysicalCamera::ImuData> &accel);


    static inline std::string CRLSourceToRosImageEncodingString(CRLCameraDataType type){
        std::string encoding;
        switch (type) {
            case CRL_GRAYSCALE_IMAGE:
                encoding = "mono8";
                break;
            case CRL_DISPARITY_IMAGE:
                encoding = "mono16";
                break;
            case CRL_COLOR_IMAGE_YUV420:
                encoding = "rgb8"; // Make sure the camera streams are also converted from yuv420 to rgb8
                break;
            default:
                Log::Logger::getInstance()->info("CRLSourceToRosImageEncodingString: encoding not specified for this CRLCameraDataType");
                break;
        }
        return encoding;
    }
};


#endif //MULTISENSE_VIEWER_RECORDFRAMES_H
