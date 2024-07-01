/**
 * @file: MultiSense-Viewer/include/Viewer/CRLCamera/MultiSense.h
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
 *   2022-3-1, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_CRLPHYSICALCAMERA_H
#define MULTISENSE_CRLPHYSICALCAMERA_H

#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <bitset>
#include <iostream>
#include <cstdint>
#include <utility>
#include <memory>
#include <glm/ext/matrix_float4x4.hpp>

#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Tools/Utils.h"
#include "CommonHeader.h"

#include <MultiSense/MultiSenseChannel.hh>


namespace VkRender::MultiSense {

    class ImageBufferWrapper {
    public:
        ImageBufferWrapper(crl::multisense::Channel *driver,
                           crl::multisense::image::Header data) :
                driver_(driver),
                callbackBuffer_(driver->reserveCallbackBuffer()),
                data_(std::move(data)) {
        }

        ~ImageBufferWrapper() {
            if (driver_) {
                driver_->releaseCallbackBuffer(callbackBuffer_);
            }
        }

        [[nodiscard]] const crl::multisense::image::Header &data() const noexcept {
            return data_;
        }

        ImageBufferWrapper operator=(const ImageBufferWrapper &) = delete;

    private:
        crl::multisense::Channel *driver_ = nullptr;
        void *callbackBuffer_;
        const crl::multisense::image::Header data_;
    };

    class IMUBufferWrapper {
    public:
        IMUBufferWrapper(crl::multisense::Channel *driver,
                         crl::multisense::imu::Header data) :
                driver_(driver),
                callbackBuffer_(driver->reserveCallbackBuffer()),
                data_(std::move(data)) {
        }

        ~IMUBufferWrapper() {
            if (driver_) {
                driver_->releaseCallbackBuffer(callbackBuffer_);
            }
        }

        [[nodiscard]] const crl::multisense::imu::Header &data() const noexcept {
            return data_;
        }

        IMUBufferWrapper operator=(const ImageBufferWrapper &) = delete;

    private:
        crl::multisense::Channel *driver_ = nullptr;
        void *callbackBuffer_;
        const crl::multisense::imu::Header data_;
    };

    class ImageBuffer {
    public:
        explicit ImageBuffer(crl::multisense::RemoteHeadChannel remoteHeadChannel, bool logSkippedFrames) :
                id(remoteHeadChannel), m_SkipLogging(logSkippedFrames) {}


        void updateImageBuffer(const std::shared_ptr<ImageBufferWrapper> &buf) {
            // Lock
            // replace latest data into m_Image pointers
            if (imagePointersMap.empty())
                return;

            if (id < crl::multisense::Remote_Head_VPB || id > crl::multisense::Remote_Head_3)
                return;

            std::scoped_lock<std::mutex> lock(mut);

            if (!m_SkipLogging) {
                if (buf->data().frameId != (counterMap[id][buf->data().source] + 1) &&
                    counterMap[id][buf->data().source] != 0) {
                    Log::Logger::getInstance()->info("We skipped frames. new frame {}, last received {}",
                                                     buf->data().frameId, (counterMap[id][buf->data().source]));
                }
            }


            imagePointersMap[id][buf->data().source] = buf;
            counterMap[id][buf->data().source] = buf->data().frameId;


        }

        // Question: making it a return statement initiates a copy? Pass by reference and return m_Image pointer?
        std::shared_ptr<ImageBufferWrapper> getImageBuffer(uint32_t idx, crl::multisense::DataSource src) {
            std::lock_guard<std::mutex> lock(mut);
            return imagePointersMap[idx][src];
        }

        crl::multisense::RemoteHeadChannel id{};
        bool m_SkipLogging = false;
    private:
        std::mutex mut;
        std::unordered_map<uint32_t, std::unordered_map<crl::multisense::DataSource, std::shared_ptr<ImageBufferWrapper>>> imagePointersMap{};
        std::unordered_map<crl::multisense::RemoteHeadChannel, std::unordered_map<crl::multisense::DataSource, int64_t>> counterMap;


    };

    class IMUBuffer {
    public:
        explicit IMUBuffer(crl::multisense::RemoteHeadChannel remoteHeadChannel, bool logSkippedFrames) :
                id(remoteHeadChannel), m_SkipLogging(logSkippedFrames) {}


        void updateIMUBuffer(const std::shared_ptr<IMUBufferWrapper> &buf) {

            // replace latest data into m_Image pointers
            if (imuPointersMap.empty())
                return;
            if (id < crl::multisense::Remote_Head_VPB || id > crl::multisense::Remote_Head_3)
                return;
            // Lock
            std::scoped_lock<std::mutex> lock(mut);

            imuPointersMap[id] = buf;
            counterMap[id] = buf->data().sequence;

        }

        // Question: making it a return statement initiates a copy? Pass by reference and return m_Image pointer?
        std::shared_ptr<IMUBufferWrapper> getIMUBuffer(uint32_t idx) {
            std::lock_guard<std::mutex> lock(mut);
            return imuPointersMap[idx];
        }

        crl::multisense::RemoteHeadChannel id{};
        bool m_SkipLogging = false;
    private:
        std::mutex mut;
        std::unordered_map<uint32_t, std::shared_ptr<IMUBufferWrapper>> imuPointersMap{};
        std::unordered_map<crl::multisense::RemoteHeadChannel, uint32_t> counterMap;


    };

    class ChannelWrapper {
    public:
        explicit ChannelWrapper(const std::string &ipAddress,
                                crl::multisense::RemoteHeadChannel remoteHeadChannel = -1, std::string ifName = "") {
#ifdef __linux__
            channelPtr_ = crl::multisense::Channel::Create(ipAddress, remoteHeadChannel, ifName);
#else
            channelPtr_ = crl::multisense::Channel::Create(ipAddress);
#endif

            bool skipLogging = false;
            // Don't log skipped frames on remote head, as we do intentionally skip frames there with the multiplexer
            if (channelPtr_) {
                crl::multisense::system::DeviceInfo deviceInfo;
                channelPtr_->getDeviceInfo(deviceInfo);
                for (std::vector v{crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_REMOTE_HEAD_VPB,
                                   crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_REMOTE_HEAD_STEREO,
                                   crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_REMOTE_HEAD_MONOCAM};
                     auto &e : v){
                    if (deviceInfo.hardwareRevision == e)
                        skipLogging = true;
                }


                //skipLogging =
            }
            imageBuffer = new ImageBuffer(remoteHeadChannel == -1 ? 0 : remoteHeadChannel, skipLogging);
            imuBuffer = new IMUBuffer(remoteHeadChannel == -1 ? 0 : remoteHeadChannel, skipLogging);

        }

        ~ChannelWrapper() {
            delete imageBuffer;
            delete imuBuffer;
            if (channelPtr_) {
                crl::multisense::Channel::Destroy(channelPtr_);
            }
            channelPtr_ = nullptr;
        }

        crl::multisense::Channel *ptr() noexcept {
            return channelPtr_;
        }

        ImageBuffer *imageBuffer{};
        IMUBuffer *imuBuffer{};

        ChannelWrapper(const ChannelWrapper &) = delete;

        ChannelWrapper operator=(const ChannelWrapper &) = delete;

    private:
        crl::multisense::Channel *channelPtr_ = nullptr;
    };


    /**
     * @brief Links LibMultiSense calls to be used with VkRender
     */
    class LibMultiSenseConnector {
    public:
        struct ImuData {
            float x, y, z;
            double time;
            double dTime;

            bool operator==(const ImuData &d) {
                if (d.x == x && d.y == y && d.z == z)
                    return true;
                else
                    return false;
            }
        };

        /**
         * @brief container to keep \refitem crl::multisense::Channel information. This block is closely related to the \refitem Parameters UI block
         */
        struct CameraInfo {
            crl::multisense::system::DeviceInfo devInfo{};
            crl::multisense::image::Config imgConf{};
            crl::multisense::image::AuxConfig auxImgConf{};
            crl::multisense::lighting::Config lightConf{};
            crl::multisense::system::NetworkConfig netConfig{};
            crl::multisense::system::VersionInfo versionInfo{};
            std::vector<crl::multisense::system::DeviceMode> supportedDeviceModes{};
            crl::multisense::DataSource supportedSources{0};
            std::vector<uint8_t *> rawImages{};
            int sensorMTU = 0;
            crl::multisense::image::Calibration calibration{};
            glm::mat4 QMat{};
            glm::mat4 KColorMat{};
            glm::mat4 KColorMatExtrinsic{};
            float focalLength = 0.0f;
            float pointCloudScale = 1.0f;

            // IMU
            std::vector<crl::multisense::imu::Info> imuSensorInfos;
            std::vector<crl::multisense::imu::Config> imuSensorConfigs;
            uint32_t imuMaxTableIndex = 0;
            bool hasIMUSensor = false;
        };


        LibMultiSenseConnector() = default;

        ~LibMultiSenseConnector() = default;

        void connect(std::string ip, std::string adapterName);

        /**@brief Starts the desired stream if supported
         *
         * @param[in] stringSrc source described in string (also shown in UI)
         * @param[in] remoteHeadID which remote head to start a stream. ID of 0 can also be a non-remotehead/ VkRender m_Device
         * @return If the requested stream was started
         */
        bool start(const std::string &stringSrc, crl::multisense::RemoteHeadChannel remoteHeadID);

        /**@brief Stops the desired stream.
         *
        * @param[in] stringSrc source described in string (also shown in UI)
        * @param[in] remoteHeadID which remote head to stop a stream. ID of 0 can also be a non-remotehead/ VkRender m_Device
        * @return If the requested stream was stopped. Returns true also if the stream was never started.
        */
        bool stop(const std::string &stringSrc, crl::multisense::RemoteHeadChannel remoteHeadID);


        /**
         *
         * @param[in] channelID Which channel to get IMU data for
         * @param[in] gyro vector to fill with measurements
         * @param[in] accel vector to fill with measurements
         * @return true fetched successfully otherwise false
         */
        bool getIMUData(crl::multisense::RemoteHeadChannel channelID, std::vector<ImuData> *gyro,
                        std::vector<ImuData> *accel) const;


        /**
         * @brief get a status update from the MultiSense m_Device
         * @param[in] channelID Which channel to fetch status for
         * @param[out] msg Parameter to be filled
         * @return
         */
        bool getStatus(crl::multisense::RemoteHeadChannel channelID, crl::multisense::system::StatusMessage *msg);


        /**@brief Sets the desired stereo filter strength
         *
         * @param[in] filterStrength Value to set
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
        */
        bool setPostFilterStrength(float filterStrength, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Sets the desired gamma value to the VkRender m_Device
         *
         * @param[in] gamma Value to set
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
        */
        bool setGamma(float gamma, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Sets the desired framerate of the sensor
         *
         * @param[in] fps Value to set
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
        */
        bool setFps(float fps, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Sets the desired gain
         *
         * @param[in] gain Value to set
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
        */
        bool setGain(float gain, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Enable or disable HDR
         *
         * @param[in] hdr True = Enable // False = Disable
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
        */
        bool setHDR(bool hdr, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Configure the sensor MTU
         *
         * @param[in] mtu Which value to set
         * @return if the value was successfully set
        */
        bool setMtu(uint32_t mtu, crl::multisense::RemoteHeadChannel channelID);

        /**@brief Get a struct of the current camera settings \ref CameraInfo
         *
         * @param[in] idx Which remote head to select
         * @return camera settings for remote head no: 'idx'
         */
        CameraInfo getCameraInfo(crl::multisense::RemoteHeadChannel idx) const;

        /**
         * @brief Sets the sensor calibration using a multisense calibration type
         * @param[in] channelID id of remote head
         * @param[in] intrinsicsFile what calibration to set
         * @param[in] intrinsicsFile what calibration to set
         * @return true successful
         */
        bool setSensorCalibration(const std::string &intrinsicsFile, const std::string &extrinsicsFile,
                                  crl::multisense::RemoteHeadChannel channelID);

        /**
         * @brief saves the sensor calibration to .yml files
         * @param[in] channelID id of remote head
         * @param[in] savePath where to save .yml files
         * @return true successful
         */
        bool saveSensorCalibration(const std::string &savePath, crl::multisense::RemoteHeadChannel channelID);

        bool getExposure(short i, bool b);

        void setIMUConfig(uint32_t tableIndex, crl::multisense::RemoteHeadChannel channelID = 0);

        MultiSenseConnectionState connectionState() {

            if (m_channelMutex.try_lock()) {
                m_channelMutex.unlock();
                return state;
            } else {
                return MULTISENSE_CHANNEL_BUSY;
            }
        }

    private:
        std::unique_ptr<ChannelWrapper> m_channel{};
        CameraInfo m_channelInfo{};
        std::mutex m_channelMutex;
        MultiSenseConnectionState state;

        /**@brief Boolean to ensure the streamcallbacks called from LibMultiSense threads dont access class data while this class is being destroyed. It does happens once in a while */
        void addCallbacks(crl::multisense::RemoteHeadChannel idx);

        static void remoteHeadCallback(const crl::multisense::image::Header &header, void *userDataP);

        static void imuCallback(const crl::multisense::imu::Header &header, void *userDataP);


        std::ostream &
        writeImageIntrinics(std::ostream &stream, const crl::multisense::image::Calibration &calibration,
                            bool hasAuxCamera);

        std::ostream &
        writeImageExtrinics(std::ostream &stream, const crl::multisense::image::Calibration &calibration,
                            bool hasAuxCamera);

        // For pointclouds
        void updateQMatrix();

        // Helper function to update and log status
        template<typename Func, typename Data>
        bool updateAndLog(Func f, Data &data, const std::string &field) {
            crl::multisense::Status status = f(data);
            if (status != crl::multisense::Status_Ok) {
                Log::Logger::getInstance()->error("Failed to update '{}'. error {}", field,
                                                  crl::multisense::Channel::statusString(status));
                return false;
            }
            return true;
        }


        void getIMUConfig(crl::multisense::RemoteHeadChannel channelID);

        void updateCameraInfo();
    };


}
#endif //MULTISENSE_CRLPHYSICALCAMERA_H
