//
// Created by magnus on 4/9/23.
//

#include <random>
#include "Viewer/Core/RendererConfig.h"
#include "Viewer/Tools/Utils.h"
#ifdef WIN32
#include "windows.h"
#endif

namespace VkRender {

#ifdef WIN32
    void getOSVersion() {
                     OSVERSIONINFOEX osvi;
            ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
            osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);

            if (GetVersionEx((LPOSVERSIONINFO)&osvi)) {
                std::cout << "Windows OS version: " << osvi.dwMajorVersion << "." << osvi.dwMinorVersion << std::endl;
            } else {
                std::cerr << "Error getting Windows OS version" << std::endl;
            }
            }
#else

    void RendererConfig::getOSVersion() {
        // OS type
        // OS version
        std::ifstream ifs("/etc/os-release");
        std::string line;
        std::string version = "empty";
        std::string name = "empty";
        bool versionFound = false, nameFound = false;
        while (std::getline(ifs, line)) {
            if (line.find("NAME=") != std::string::npos) {
                name = line.substr(line.find("=") + 1);
                name.erase(std::remove(name.begin(), name.end(), '"'), name.end());
                nameFound = true;
            }
            if (line.find("VERSION=") != std::string::npos) {
                version = line.substr(line.find("=") + 1);
                version.erase(std::remove(version.begin(), version.end(), '"'), version.end());
                versionFound = true;
            }
            if (versionFound && nameFound)
                break;
        }
        Log::Logger::getInstance()->info("Found Ubuntu version: {}", version);
        m_OS = name;
        m_OSVersion = version;
    }

    std::string RendererConfig::fetchArchitecture() {
#ifdef WIN32
#else
        struct utsname unameData{};
        if (uname(&unameData) != 0) {
            Log::Logger::getInstance()->error("Error calling uname()", unameData.machine);
            return "Unable to read";
        }
        Log::Logger::getInstance()->info("Found architecture: {}", unameData.machine);
        return unameData.machine;
#endif
    }

    std::string RendererConfig::fetchApplicationVersion() {
        auto path = Utils::getAssetsPath() / "Generated/VersionInfo";
        std::ifstream infile(path);
        std::string line;
        std::string applicationVersion = "Not found";

        while (std::getline(infile, line)) {
            if (line.find("VERSION=") != std::string::npos) {
                applicationVersion = line.substr(line.find("=") + 1);
                applicationVersion.erase(std::remove(applicationVersion.begin(), applicationVersion.end(), '"'), applicationVersion.end());
            }
            if (line.find("SERVER=") != std::string::npos) {
                m_ServerInfo.server = line.substr(line.find("=") + 1);
                m_ServerInfo.server.erase(std::remove(m_ServerInfo.server.begin(), m_ServerInfo.server.end(), '"'), m_ServerInfo.server.end());
            }
            if (line.find("PROTOCOL=") != std::string::npos) {
                m_ServerInfo.protocol = line.substr(line.find("=") + 1);
                m_ServerInfo.protocol.erase(std::remove(m_ServerInfo.protocol.begin(), m_ServerInfo.protocol.end(), '"'), m_ServerInfo.protocol.end());
            }
            if (line.find("DESTINATION=") != std::string::npos) {
                m_ServerInfo.destination = line.substr(line.find("=") + 1);
                m_ServerInfo.destination.erase(std::remove(m_ServerInfo.destination.begin(), m_ServerInfo.destination.end(), '"'), m_ServerInfo.destination.end());
            }

        }
        Log::Logger::getInstance()->info("Found Application version: {}", applicationVersion);
        return applicationVersion;
    }

    const std::string &RendererConfig::getArchitecture() const {
        return m_Architecture;
    }

    const std::string &RendererConfig::getOsVersion() const {
        return m_OSVersion;
    }

    const std::string &RendererConfig::getAppVersion() const {
        return m_AppVersion;
    }

    const std::string &RendererConfig::getTimeStamp() const {
        return m_TimeStamp;
    }

    const std::string &RendererConfig::getOS() const {
        return m_OS;
    }

    const std::string &RendererConfig::getGpuDevice() const {
        return m_GPUDevice;
    }

    void RendererConfig::setGpuDevice(VkPhysicalDevice const &physicalDevice) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        m_GPUDevice = properties.deviceName;
    }

    const std::string &RendererConfig::getAnonIdentifierString() const {
        return m_Identifier;
    }


#endif
};