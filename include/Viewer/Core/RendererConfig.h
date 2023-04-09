//
// Created by magnus on 4/9/23.
//

#ifndef MULTISENSE_VIEWER_RENDERCONFIG_H
#define MULTISENSE_VIEWER_RENDERCONFIG_H

#include <string>
#include <fstream>
#include <vulkan/vulkan.hpp>

#ifdef WIN32

#else
#include <sys/utsname.h>
#endif

#include "Viewer/Tools/Logger.h"

namespace VkRender {
    class RendererConfig {
    public:
        static RendererConfig &getInstance() {
            static RendererConfig instance;
            return instance;
        }

        const std::string &getArchitecture() const;

        const std::string &getOsVersion() const;

        const std::string &getAppVersion() const;

        const std::string &getTimeStamp() const;

        const std::string &getOS() const;

        const std::string &getGpuDevice() const;

        void setGpuDevice(const VkPhysicalDevice& physicalDevice);

    private:
        RendererConfig() {
            getOSVersion();
            m_Architecture = fetchArchitecture();
            m_AppVersion = fetchApplicationVersion();
        }




        std::string m_Architecture;
        std::string m_OSVersion;
        std::string m_OS;
        std::string m_AppVersion;
        std::string m_TimeStamp;
        std::string m_GPUDevice;

        void getOSVersion();
        std::string fetchArchitecture();
        std::string fetchApplicationVersion();
    };

};
#endif //MULTISENSE_VIEWER_RENDERCONFIG_H
