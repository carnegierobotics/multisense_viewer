//
// Created by mgjer on 12/09/2024.
//

#ifndef MULTISENSE_VIEWER_WRAPPER_H
#define MULTISENSE_VIEWER_WRAPPER_H

#include <MultiSense/MultiSenseChannel.hh>
#include <MultiSense/MultiSenseTypes.hh>

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
    const crl::multisense::image::Header data_{};
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
    explicit ImageBuffer(bool logSkippedFrames) :
            m_SkipLogging(logSkippedFrames) {}


    void updateImageBuffer(const std::shared_ptr<ImageBufferWrapper> &buf) {
        // Lock
        // replace latest data into m_Image pointers
        if (imagePointersMap.empty())
            return;

        std::scoped_lock<std::mutex> lock(mut);

        if (!m_SkipLogging) {
            if (buf->data().frameId != (counterMap[buf->data().source] + 1) &&
                counterMap[buf->data().source] != 0) {
                Log::Logger::getInstance()->info("We skipped frames. new frame {}, last received {}",
                                                 buf->data().frameId, (counterMap[buf->data().source]));
            }
        }
        imagePointersMap[buf->data().source] = buf;
        counterMap[buf->data().source] = buf->data().frameId;
    }

    // Question: making it a return statement initiates a copy? Pass by reference and return m_Image pointer?
    std::shared_ptr<ImageBufferWrapper> getImageBuffer(crl::multisense::DataSource src) {
        std::lock_guard<std::mutex> lock(mut);
        return imagePointersMap[src];
    }

    bool m_SkipLogging = false;
private:
    std::mutex mut{};
    std::unordered_map<crl::multisense::DataSource, std::shared_ptr<ImageBufferWrapper>> imagePointersMap{};
    std::unordered_map<crl::multisense::DataSource, int64_t> counterMap{};


};

class IMUBuffer {
public:
    explicit IMUBuffer(bool logSkippedFrames) : m_SkipLogging(logSkippedFrames) {}

    void updateIMUBuffer(const std::shared_ptr<IMUBufferWrapper> &buf) {
        // Lock
        std::scoped_lock<std::mutex> lock(mut);
        imuPointersMap = buf;
        counterMap = buf->data().sequence;
    }

    // Question: making it a return statement initiates a copy? Pass by reference and return m_Image pointer?
    std::shared_ptr<IMUBufferWrapper> getIMUBuffer() {
        std::lock_guard<std::mutex> lock(mut);
        return imuPointersMap;
    }

    bool m_SkipLogging = false;
private:
    std::mutex mut{};
    std::shared_ptr<IMUBufferWrapper> imuPointersMap{};
    uint32_t counterMap{};


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
        imageBuffer = new ImageBuffer(skipLogging);
        imuBuffer = new IMUBuffer(skipLogging);

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

#endif //MULTISENSE_VIEWER_WARPPER_H
