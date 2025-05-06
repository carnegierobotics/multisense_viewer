// SYCLDeviceSelector.h
#ifndef MULTISENSE_VIEWER_SYCLDEVICESELECTOR_H
#define MULTISENSE_VIEWER_SYCLDEVICESELECTOR_H

#include <memory>
#include <mutex>
#include <map>
#include "Viewer/Tools/Logger.h"


#ifdef SYCL_ENABLED
#include <sycl/sycl.hpp>
#endif

namespace VkRender {
    enum class SYCLDeviceType {
        CPU,
        GPU,
        Default
    };

    static std::string syclDeviceTypeToString(SYCLDeviceType type) {
        switch (type) {
        case SYCLDeviceType::GPU: return "GPU";
        case SYCLDeviceType::CPU: return "CPU";
        case SYCLDeviceType::Default: return "Default";
        default: throw std::invalid_argument("Unknown SYCLDeviceType");
        }
    }

    static SYCLDeviceType stringToSyclDeviceType(const std::string& str) {
        std::string lowercase = str;
        std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(), ::tolower);

        if (lowercase == "gpu") return SYCLDeviceType::GPU;
        if (lowercase == "cpu") return SYCLDeviceType::CPU;
        if (lowercase == "default") return SYCLDeviceType::Default;

        throw std::invalid_argument("Invalid SYCLDeviceType string: " + str);
    }




#ifdef SYCL_ENABLED
    class SYCLDeviceSelector {
    public:
        SYCLDeviceSelector() = delete;
        explicit SYCLDeviceSelector(SYCLDeviceType deviceType);
        ~SYCLDeviceSelector();

        sycl::queue &getQueue();

        [[nodiscard]] const sycl::device&   getSyclDevice()   const { return m_device; }
        [[nodiscard]] std::string           getDeviceName()   const {
            return m_device.get_info<sycl::info::device::name>();
        }
        [[nodiscard]] std::string           getPlatformName() const {
            return m_device.get_platform().get_info<sycl::info::platform::name>();
        }
        [[nodiscard]] std::string           getPlatformVendor() const {
            return m_device.get_platform().get_info<sycl::info::platform::vendor>();
        }

        [[nodiscard]] bool isDeviceAvailable() const {return m_isDeviceTypeAvailable;}

        [[nodiscard]] SYCLDeviceType getDeviceType() const { return m_deviceType; }
    private:
        sycl::queue m_queue;
        sycl::device m_device;
        bool m_isDeviceTypeAvailable = false;
        SYCLDeviceType m_deviceType;

        bool selectDevice(SYCLDeviceType deviceType);
    };

    class SYCLDeviceManager {
    public:
        SYCLDeviceManager();
        static SYCLDeviceManager &getInstance();
        std::shared_ptr<SYCLDeviceSelector> getDevice(SYCLDeviceType type);

        SYCLDeviceManager(const SYCLDeviceManager &) = delete;
        SYCLDeviceManager &operator=(const SYCLDeviceManager &) = delete;

    private:
        std::mutex m_mutex;
        std::map<SYCLDeviceType, std::shared_ptr<SYCLDeviceSelector>> m_devices;
    };

#else
    class SYCLDeviceSelector {
    public:
        SYCLDeviceSelector() = delete;
        explicit SYCLDeviceSelector(SYCLDeviceType deviceType);
        ~SYCLDeviceSelector();

    private:
        bool selectDevice(SYCLDeviceType deviceType);
    };

    class SYCLDeviceManager {
    public:
        SYCLDeviceManager() = default;
        static SYCLDeviceManager &getInstance();
        std::shared_ptr<SYCLDeviceSelector> getDevice(SYCLDeviceType type);

        SYCLDeviceManager(const SYCLDeviceManager &) = delete;
        SYCLDeviceManager &operator=(const SYCLDeviceManager &) = delete;

    private:
        std::mutex m_mutex;
        std::map<SYCLDeviceType, std::shared_ptr<SYCLDeviceSelector>> m_devices;
    };

#endif

}

#endif //MULTISENSE_VIEWER_SYCLDEVICESELECTOR_H
