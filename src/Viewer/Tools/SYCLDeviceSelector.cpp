//
// Created by magnus-desktop on 4/14/25.
//
// SYCLDeviceSelector.cpp
#ifdef SYCL_ENABLED

#include "Viewer/Tools/SYCLDeviceSelector.h"

namespace VkRender {

    SYCLDeviceSelector::SYCLDeviceSelector(SYCLDeviceType deviceType) {
        selectDevice(deviceType);
    }

    SYCLDeviceSelector::~SYCLDeviceSelector() {
        try {
            Log::Logger::getInstance()->info("Destroying Sycl Queue for device: {}",
                                             m_queue.get_device().get_info<sycl::info::device::name>());
            m_queue.wait();
        }
        catch (const std::exception &e) {
            Log::Logger::getInstance()->error("Error waiting on SYCL queue: {}", e.what());
        }
    }

    sycl::queue &SYCLDeviceSelector::getQueue() {
        return m_queue;
    }

    bool SYCLDeviceSelector::selectDevice(SYCLDeviceType deviceType) {
        auto error_handler = [](sycl::exception_list el) {
            for (auto &e : el) {
                try {
                    std::rethrow_exception(e);
                } catch (const sycl::exception &ex) {
                    Log::Logger::getInstance()->error("Asynchronous SYCL exception: {}", ex.what());
                    return false;
                }
            }
            return true;
        };

        sycl::property_list properties{sycl::property::queue::in_order{}};

        try {
            if (deviceType == SYCLDeviceType::GPU) {
                m_queue = sycl::queue(sycl::gpu_selector_v, error_handler, properties);
                Log::Logger::getInstance()->info("Using GPU: {}",
                    m_queue.get_device().get_info<sycl::info::device::name>());
            } else if (deviceType == SYCLDeviceType::CPU) {
                m_queue = sycl::queue(sycl::cpu_selector_v, error_handler, properties);
                Log::Logger::getInstance()->info("Using CPU: {}",
                    m_queue.get_device().get_info<sycl::info::device::name>());
            } else {
                m_queue = sycl::queue(sycl::default_selector_v, error_handler, properties);
                Log::Logger::getInstance()->info("Using default device: {}",
                    m_queue.get_device().get_info<sycl::info::device::name>());
            }
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->error("Error selecting device: {}", e.what());
            Log::Logger::getInstance()->error("Falling back to default device.");
            m_queue = sycl::queue(sycl::default_selector_v, error_handler, properties);
            Log::Logger::getInstance()->info("Using device: {}",
                m_queue.get_device().get_info<sycl::info::device::name>());
        }

        try {
            m_queue.submit([&](sycl::handler &cgh) {
                cgh.single_task<class DummyKernel>([]() {});
            }).wait();
            Log::Logger::getInstance()->info("Dummy kernel launched successfully on device: {}",
                m_queue.get_device().get_info<sycl::info::device::name>());
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->error("Error launching dummy kernel on device {}: {}",
                m_queue.get_device().get_info<sycl::info::device::name>(), e.what());
            Log::Logger::getInstance()->info("Falling back to CPU device.");
            m_queue = sycl::queue(sycl::cpu_selector_v, error_handler, properties);
            m_queue.submit([&](sycl::handler &cgh) {
                cgh.single_task<class DummyKernelCPU>([]() {});
            }).wait_and_throw();
        }

        return true;
    }

    SYCLDeviceManager &SYCLDeviceManager::getInstance() {
        static SYCLDeviceManager instance;
        return instance;
    }

    SYCLDeviceManager::SYCLDeviceManager() {
        Log::Logger::getInstance()->info("Creating SYCLDeviceManager");
        m_devices[SYCLDeviceType::GPU] = std::make_shared<SYCLDeviceSelector>(SYCLDeviceType::GPU);
        m_devices[SYCLDeviceType::CPU] = std::make_shared<SYCLDeviceSelector>(SYCLDeviceType::CPU);
        m_devices[SYCLDeviceType::Default] = std::make_shared<SYCLDeviceSelector>(SYCLDeviceType::Default);
    }

    std::shared_ptr<SYCLDeviceSelector> SYCLDeviceManager::getDevice(SYCLDeviceType type) {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_devices.at(type);
    }


} // namespace VkRender

#endif // SYCL_ENABLED
