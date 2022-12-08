//
// Created by magnus on 3/1/22.
//

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
#include <MultiSense/MultiSenseChannel.hh>

#include "Viewer/Core/Definitions.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Scripts/Private/TextureDataDef.h"


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

    class ImageBuffer {
    public:
        explicit ImageBuffer(crl::multisense::RemoteHeadChannel remoteHeadChannel, bool logSkippedFrames) :
                id(remoteHeadChannel), m_SkipLogging(logSkippedFrames) {}


        void updateImageBuffer(const std::shared_ptr<ImageBufferWrapper> &buf) {
            // Lock
            std::scoped_lock<std::mutex> lock(mut);
            // replace latest data into m_Image pointers
            if (imagePointersMap.empty())
                return;

            if (!m_SkipLogging) {
                if (buf->data().frameId != (counterMap[id][buf->data().source] + 1) &&
                    counterMap[id][buf->data().source] != 0) {
                    Log::Logger::getInstance()->info("We skipped frames. new frame {}, last received {}",
                                                     buf->data().frameId, (counterMap[id][buf->data().source]));
                }
            }


            imagePointersMap[id][buf->data().source] = buf;
            counterMap[id][buf->data().source] = buf->data().frameId;

            Log::Logger::getLogMetrics()->device.sourceReceiveMapCounter[id][Utils::dataSourceToString(
                    buf->data().source)]++;

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
        std::unordered_map<crl::multisense::RemoteHeadChannel, std::unordered_map<crl::multisense::DataSource, uint32_t>> counterMap;


    };

    class ChannelWrapper {
    public:
        explicit ChannelWrapper(const std::string &ipAddress,
                                crl::multisense::RemoteHeadChannel remoteHeadChannel = -1, std::string ifName = "") {
#ifdef __linux__
            channelPtr_ = crl::multisense::Channel::Create(ipAddress, remoteHeadChannel, ifName);
#else
            channelPtr_ = crl::multisense::Channel::Create(ipAddress, remoteHeadChannel);
#endif

            bool skipLogging = false;
            // Don't log skipped frames on remote head, as we do intentionally skip frames there
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
        }

        ~ChannelWrapper() {
            delete imageBuffer;
            // Reset image counter for debugger
            for (auto &src: Log::Logger::getLogMetrics()->device.sourceReceiveMapCounter)
                for (auto &counter: src.second)
                    counter.second = 0;


            if (channelPtr_) {
                crl::multisense::Channel::Destroy(channelPtr_);
            }
        }

        crl::multisense::Channel *ptr() noexcept {
            return channelPtr_;
        }

        ImageBuffer *imageBuffer{};

        ChannelWrapper(const ChannelWrapper &) = delete;

        ChannelWrapper operator=(const ChannelWrapper &) = delete;

    private:
        crl::multisense::Channel *channelPtr_ = nullptr;
    };


    /**
     * @brief Links LibMultiSense calls to be used with VkRender
     */
    class CRLPhysicalCamera {
    public:
        CRLPhysicalCamera() = default;

        ~CRLPhysicalCamera() = default;

        /**
         * @brief container to keep \refitem crl::multisense::Channel information. This block is closely related to the \refitem Parameters UI block
         */
        struct CameraInfo {
            crl::multisense::system::DeviceInfo devInfo{};
            crl::multisense::image::Config imgConf{};
            crl::multisense::lighting::Config lightConf{};
            crl::multisense::system::NetworkConfig netConfig{};
            crl::multisense::system::VersionInfo versionInfo{};
            std::vector<crl::multisense::system::DeviceMode> supportedDeviceModes{};
            crl::multisense::DataSource supportedSources{0};
            std::vector<uint8_t *> rawImages{};
            int sensorMTU = 0;
            crl::multisense::image::Calibration calibration{};
            glm::mat4 QMat{};
        };

        /**@brief Connects to a VkRender m_Device
         *
         * @param[in] ip Which IP the camera is located on
         * @param[in] isRemoteHead If the m_Device is a remote head or not
         * @return vector containing the list of successful connections. Numbered by crl::multisense::RemoteHeadChannel ids
         */
        std::vector<crl::multisense::RemoteHeadChannel>
        connect(const VkRender::Device *ip, bool isRemoteHead, const std::string &ifName);

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

        /**@brief Connects the MultiSense interface with the Renderer interface. Puts camera data into a \ref VkRender::TextureData objectS. Caller must ensure enough memory is allocated for the object
        *
        * @param[in] stringSrc source described in string (also shown in UI)
        * @param[out] tex Pointer to a texture data struct \refitem TextureData with pre-allocated memory
        * @param[in] remoteHeadID which remote head to start a stream. ID of 0 can also be a non-remotehead/ MultiSense m_Device
        * @return If true if a frame was copied into the 'tex' object
        */
        bool
        getCameraStream(const std::string &stringSrc, VkRender::TextureData *tex,
                        crl::multisense::RemoteHeadChannel idx);


        /**
         * @brief get a status update from the MultiSense m_Device
         * @param[in] channelID Which channel to fetch status for
         * @param[out] msg Parameter to be filled
         * @return
         */
        bool getStatus(crl::multisense::RemoteHeadChannel channelID, crl::multisense::system::StatusMessage *msg);

        /**@brief Constructs the Q matrix from the calibration data and stores it in \ref infoMap
        *
        * @param[in] width Width of desired m_Image to construct Q matrix for. Used to obtain correct scaling
        * @param[in] channelID which remote head to select
        */
        void preparePointCloud(uint32_t width, crl::multisense::RemoteHeadChannel channelID);

        /** @brief Sets the desired resolution of the camera. Must be one of supported resolutions of the sensor
         *
         * @param[in] resolution Resolution enum
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
         * */
        bool setResolution(CRLCameraResolution resolution, crl::multisense::RemoteHeadChannel channelID);

        /** @brief Sets the exposure parameters for the sensor
         *
         * @param[in] p \ref ExposureParams
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
         * */
        bool setExposureParams(ExposureParams p, crl::multisense::RemoteHeadChannel channelID);

        /** @brief Sets the white balance for the sensor
         *
         * @param[in] param \ref WhiteBalanceParams
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
         * */
        bool setWhiteBalance(WhiteBalanceParams param, crl::multisense::RemoteHeadChannel channelID);

        /** @brief Sets the desired resolution of the camera. Must be one of supported resolutions of the sensor
         *
         * @param[in] resolution Resolution enum
         * @param[in] channelID which remote head to select
         * @return if the value was successfully set
         * */
        bool setLighting(LightingParams light, crl::multisense::RemoteHeadChannel channelID);

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
        CameraInfo getCameraInfo(crl::multisense::RemoteHeadChannel idx);

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
        bool saveSensorCalibration(const std::string &savePath, short channelID);

    private:
        std::unordered_map<crl::multisense::RemoteHeadChannel, std::unique_ptr<ChannelWrapper>> channelMap{};
        std::unordered_map<crl::multisense::RemoteHeadChannel, CRLCameraResolution> currentResolutionMap{};
        std::unordered_map<crl::multisense::RemoteHeadChannel, CameraInfo> infoMap;
        std::mutex setCameraDataMutex;

        /**@brief Boolean to ensure the streamcallbacks called from LibMultiSense threads dont access class data while this class is being destroyed. It does happens once in a while */
        void addCallbacks(crl::multisense::RemoteHeadChannel idx);

        static void remoteHeadCallback(const crl::multisense::image::Header &header, void *userDataP);

        /**@brief Updates the \ref CameraInfo struct for the chosen remote head. Usually only called once on first connection
         *
         * @param[in] idx Which remote head to select
         *
         */
        void updateCameraInfo(crl::multisense::RemoteHeadChannel idx);

        std::ostream &
        writeImageIntrinics(std::ostream &stream, const crl::multisense::image::Calibration &calibration,
                            bool hasAuxCamera);

        std::ostream &
        writeImageExtrinics(std::ostream &stream, const crl::multisense::image::Calibration &calibration,
                            bool hasAuxCamera);
    };


}
#endif //MULTISENSE_CRLPHYSICALCAMERA_H
