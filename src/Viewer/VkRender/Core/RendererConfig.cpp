//
// Created by magnus on 4/9/23.
//

#include <random>
#include <nlohmann/json.hpp>

#include "Viewer/VkRender/Core/RendererConfig.h"
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/Tools/Utils.h"

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
            if (line.find("DESTINATION_VERSIONINFO=") != std::string::npos) {
                m_ServerInfo.versionInfoDestination = line.substr(line.find("=") + 1);
                m_ServerInfo.versionInfoDestination.erase(
                        std::remove(m_ServerInfo.versionInfoDestination.begin(), m_ServerInfo.versionInfoDestination.end(), '"'),
                        m_ServerInfo.versionInfoDestination.end());
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
        m_AppVersionRemote = applicationVersion;
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

    AppConfig::ApplicationUserSetting &RendererConfig::getUserSetting() {
        return m_UserSetting;
    }

    AppConfig::ApplicationUserSetting* RendererConfig::getUserSettingRef() {
        return &m_UserSetting;
    }

    const std::string &RendererConfig::getAnonymousIdentifier() const {
        return m_Identifier;
    }

    const std::string &RendererConfig::getAppVersionRemote() const {
        return m_AppVersionRemote;
    }

    void RendererConfig::setAppVersionRemote(const std::string &mAppVersionRemote) {
        m_AppVersionRemote = mAppVersionRemote;
    }

    const std::vector<std::string> &RendererConfig::getEnabledExtensions() const {
        return m_EnabledExtensions;
    }

    void RendererConfig::addEnabledExtension(const std::string& extensionName) {
        m_EnabledExtensions.emplace_back(extensionName);
    }

    bool RendererConfig::hasEnabledExtension(const std::string &extensionName) const {
        return Utils::isInVector(m_EnabledExtensions, extensionName);
    }

    void RendererConfig::saveSettings(Renderer* ctx) {
        // Save application settings to file
        // Save application settings to file
        try {

            // Save m_editors to a settings json file
            // Save type of editor and position/size, index
            Log::Logger::getInstance()->info("Attempting to save editor settings to: {}", Utils::getRuntimeConfigFilePath().string());

            nlohmann::json jsonContent;
            // Create editor settings JSON array
            nlohmann::json jsonEditors = nlohmann::json::array();

            /*
            for (auto& editor : ctx->m_editors) {
                VulkanRenderPassCreateInfo& createInfo = editor->getCreateInfo();
                nlohmann::json jsonEditor;
                jsonEditor["width"] = createInfo.width;
                jsonEditor["height"] = createInfo.height;
                jsonEditor["x"] = createInfo.x;
                jsonEditor["y"] = createInfo.y;
                jsonEditor["borderSize"] = createInfo.borderSize;
                jsonEditor["editorTypeDescription"] = createInfo.editorTypeDescription;
                jsonEditor["resizeable"] = createInfo.resizeable;
                jsonEditor["editorIndex"] = createInfo.editorIndex;
                jsonEditor["uiLayers"] = createInfo.uiLayers;

                jsonEditors.push_back(jsonEditor);
                Log::Logger::getInstance()->info("Editor {}: type = {}, x = {}, y = {}, width = {}, height = {}",
                                                 createInfo.editorIndex, editorTypeToString(createInfo.editorTypeDescription), createInfo.x, createInfo.y, createInfo.width, createInfo.height);
            }
            */
            jsonContent["editors"] = jsonEditors;

            jsonContent["generalSettings"]= m_UserSetting;

            // Path where settings will be saved
            std::filesystem::path settingsFilePath = Utils::getRuntimeConfigFilePath();

            // Write JSON to file
            std::ofstream outFile(settingsFilePath, std::ios_base::app);
            if (outFile.is_open()) {
                outFile << std::setw(4) << jsonContent << std::endl;
                outFile.close();
                Log::Logger::getInstance()->info("Saved application settings to {}", settingsFilePath.string().c_str());
            } else {
                Log::Logger::getInstance()->info("Failed to open settings file at {}", settingsFilePath.string().c_str());
            }
        } catch (const std::exception& e) {
            Log::Logger::getInstance()->error("Exception while saving settings: {}", e.what());
        }


    }

    // Convert LogLevel enum to string and vice versa
    NLOHMANN_JSON_SERIALIZE_ENUM( Log::LogLevel, {
        {Log::LOG_LEVEL_INFO, "info"},
        {Log::LOG_LEVEL_TRACE, "trace"},
        {Log::LOG_LEVEL_DEBUG, "debug"}
    })
    // Provide to_json and from_json functions


}
