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

    void RendererConfig::getOSVersion() {
        std::string major;
        std::string minor;

        OSVERSIONINFOEX osvi;
        std::memset(&osvi, 0, sizeof(OSVERSIONINFOEX));
        osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
        osvi.dwMajorVersion = 11; // Windows 11
        osvi.dwMinorVersion = 0;

        ULONGLONG conditionMask = VerSetConditionMask(0, VER_MAJORVERSION, VER_GREATER_EQUAL);
        conditionMask = VerSetConditionMask(conditionMask, VER_MINORVERSION, VER_GREATER_EQUAL);

        if (VerifyVersionInfo(&osvi, VER_MAJORVERSION | VER_MINORVERSION, conditionMask)) {
            major = std::to_string(osvi.dwMajorVersion);
            minor = std::to_string(osvi.dwMinorVersion);
            Log::Logger::getInstance()->info("Found Windows version. Running version 11 or later");

        } else {
            osvi.dwMajorVersion = 10; // Windows 10
            osvi.dwMinorVersion = 0;
            if (VerifyVersionInfo(&osvi, VER_MAJORVERSION | VER_MINORVERSION, conditionMask)) {
                major = std::to_string(osvi.dwMajorVersion);
                minor = std::to_string(osvi.dwMinorVersion);
                Log::Logger::getInstance()->info("Found Windows version. Running version 10 or later");

            } else {
                Log::Logger::getInstance()->error("Unsupported or unlicensed Windows version");
                major = minor = "0";
            }
        }


        std::string versionString = major + "." + minor;
        Log::Logger::getInstance()->info("Found Windows version: {}", versionString);
        m_OS = "Windows";
        m_OSVersion = versionString;
    }

    std::string RendererConfig::fetchArchitecture() {
        SYSTEM_INFO systemInfo;
        GetNativeSystemInfo(&systemInfo);

        if (systemInfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64) {
            Log::Logger::getInstance()->info("Found 64-bit architecture");
            return "64-bit";
        } else if (systemInfo.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_INTEL) {
            Log::Logger::getInstance()->info("Found 32-bit architecture");
            return "32-bit";
        } else {
            return "Unknown architecture";
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
        Log::Logger::getInstance()->info("Found {} version: {}", name, version);
        m_OS = name;
        m_OSVersion = version;
    }

    std::string RendererConfig::fetchArchitecture() {
        struct utsname unameData{};
        if (uname(&unameData) != 0) {
            Log::Logger::getInstance()->error("Error calling uname()", unameData.machine);
            return "Unable to read";
        }
        Log::Logger::getInstance()->info("Found architecture: {}", unameData.machine);
        return unameData.machine;
    }



#endif

    std::string RendererConfig::fetchApplicationVersion() {
        auto path = Utils::getAssetsPath() / "Generated/VersionInfo";
        std::ifstream infile(path);
        std::string line;
        std::string applicationVersion = "Not found";

        while (std::getline(infile, line)) {
            if (line.find("VERSION=") != std::string::npos) {
                applicationVersion = line.substr(line.find("=") + 1);
                applicationVersion.erase(std::remove(applicationVersion.begin(), applicationVersion.end(), '"'),
                                         applicationVersion.end());
            }
            if (line.find("SERVER=") != std::string::npos) {
                m_ServerInfo.server = line.substr(line.find("=") + 1);
                m_ServerInfo.server.erase(std::remove(m_ServerInfo.server.begin(), m_ServerInfo.server.end(), '"'),
                                          m_ServerInfo.server.end());
            }
            if (line.find("PROTOCOL=") != std::string::npos) {
                m_ServerInfo.protocol = line.substr(line.find("=") + 1);
                m_ServerInfo.protocol.erase(
                        std::remove(m_ServerInfo.protocol.begin(), m_ServerInfo.protocol.end(), '"'),
                        m_ServerInfo.protocol.end());
            }
            if (line.find("DESTINATION=") != std::string::npos) {
                m_ServerInfo.destination = line.substr(line.find("=") + 1);
                m_ServerInfo.destination.erase(
                        std::remove(m_ServerInfo.destination.begin(), m_ServerInfo.destination.end(), '"'),
                        m_ServerInfo.destination.end());
            }
            if (line.find("LOG_LEVEL=") != std::string::npos) {
                std::string logLevelStr = line.substr(line.find("=") + 1);
                logLevelStr.erase(
                        std::remove(logLevelStr.begin(), logLevelStr.end(), '"'),
                        logLevelStr.end());
               m_UserSetting.logLevel = Utils::getLogLevelEnumFromString(logLevelStr);
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

    const RendererConfig::ApplicationUserSetting &RendererConfig::getUserSetting() const {
        return m_UserSetting;
    }

    RendererConfig::ApplicationUserSetting* RendererConfig::getUserSettingRef() {
        return &m_UserSetting;
    }

};