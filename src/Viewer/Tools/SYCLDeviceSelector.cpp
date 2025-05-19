//
// Created by magnus-desktop on 4/14/25.
//
// SYCLDeviceSelector.cpp
#ifdef SYCL_ENABLED

#include "Viewer/Tools/SYCLDeviceSelector.h"

namespace VkRender {




    SYCLDeviceSelector::SYCLDeviceSelector(SYCLDeviceType deviceType) : m_deviceType(deviceType) {
        m_isDeviceTypeAvailable = selectDevice(deviceType);

    }

    SYCLDeviceSelector::~SYCLDeviceSelector() {
        try {
            auto dev = m_queue.get_device();
            Log::Logger::getInstance()->info(
                "Destroying SYCL queue (device: {})",
                dev.get_info<sycl::info::device::name>()
            );
            m_queue.wait();
        } catch (const std::exception &e) {
            Log::Logger::getInstance()->error(
                "Error when waiting on SYCL queue destruction: {}",
                e.what()
            );
        }
    }

    sycl::queue &SYCLDeviceSelector::getQueue() {
        return m_queue;
    }

    bool SYCLDeviceSelector::selectDevice(SYCLDeviceType deviceType) {
        // 1) Pick the device via the proper selector:
        try {
            switch (deviceType) {
                case SYCLDeviceType::GPU:
                    m_device = sycl::device(sycl::gpu_selector_v);
                    break;
                case SYCLDeviceType::CPU:
                    m_device = sycl::device(sycl::cpu_selector_v);
                    break;
                default:
                    m_device = sycl::device(sycl::default_selector_v);
            }
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->error(
                "SYCL device selection failed: {} — falling back to default.",
                e.what()
            );
            m_device = sycl::device(sycl::default_selector_v);
        }

        // Log what we got
        Log::Logger::getInstance()->info(
            "Selected device: {} [{}]",
            m_device.get_info<sycl::info::device::name>(),
            m_device.is_gpu() ? "GPU" : m_device.is_cpu() ? "CPU" : m_device.is_host() ? "Host" : "Other"
        );

        // 2) Build a queue on that device, with in-order property + async handler:
        auto error_handler = [](sycl::exception_list exList) {
            for (auto &e: exList) {
                try { std::rethrow_exception(e); } catch (const sycl::exception &ex) {
                    Log::Logger::getInstance()->error(
                        "Asynchronous SYCL exception: {}", ex.what()
                    );
                }
            }
        };
        sycl::property_list props{sycl::property::queue::in_order{}};

        try {
            m_queue = sycl::queue(m_device, error_handler, props);
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->error(
                "Queue construction failed on {}: {} — trying default device queue.",
                m_device.get_info<sycl::info::device::name>(),
                e.what()
            );
            m_device = sycl::device(sycl::default_selector_v);
            m_queue = sycl::queue(m_device, error_handler, props);
        }

        // 3) Run a tiny "dummy" kernel to verify everything works:
        try {
            m_queue.submit([&](sycl::handler &cgh) {
                cgh.single_task<class DummyKernelTest>([]() {
                });
            }).wait_and_throw();

            Log::Logger::getInstance()->info(
                "Dummy kernel succeeded on {}",
                m_device.get_info<sycl::info::device::name>()
            );
        } catch (const std::bad_function_call &e) {
            Log::Logger::getInstance()->error(
                "Dummy kernel failed on {}: {} — falling back to CPU queue.",
                m_device.get_info<sycl::info::device::name>(),
                e.what()
            );
            m_device = sycl::device(sycl::cpu_selector_v);
            m_queue = sycl::queue(m_device, error_handler, props);
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->error(
                "Dummy kernel failed on {}: {} — falling back to CPU queue.",
                m_device.get_info<sycl::info::device::name>(),
                e.what()
            );
            m_device = sycl::device(sycl::cpu_selector_v);
            m_queue = sycl::queue(m_device, error_handler, props);
        }

        if (deviceType == SYCLDeviceType::CPU) {
            return m_device.is_cpu();
        }
        if (deviceType == SYCLDeviceType::GPU) {
            return m_device.is_gpu();
        }
        // for Default, consider both CPU or GPU valid:
        return m_device.is_cpu() || m_device.is_gpu();
    }

    SYCLDeviceManager &SYCLDeviceManager::getInstance() {
        static SYCLDeviceManager instance;
        return instance;
    }

    SYCLDeviceManager::SYCLDeviceManager() {
        cleanupAdaptiveCppCache("MultiSense-Viewer"); // TODO A more failsafe way of cleaning up adaptivecpp cache
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
