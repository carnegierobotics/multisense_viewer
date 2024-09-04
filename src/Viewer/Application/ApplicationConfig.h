//
// Created by magnus on 4/9/23.
//

#ifndef MULTISENSE_VIEWER_RENDERCONFIG_H
#define MULTISENSE_VIEWER_RENDERCONFIG_H

#include "pch.h"
#include <vulkan/vulkan.hpp>


#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <winsock2.h>
#include <iphlpapi.h>
#include <cstdlib>
#pragma comment(lib, "IPHLPAPI.lib")
#ifdef APIENTRY
#undef APIENTRY
#endif
#else

#include <sys/utsname.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include <netinet/in.h>
#include <string.h>

#endif

#include "Viewer/Tools/Logger.h"
#include "Viewer/Tools/Utils.h"
#include "ApplicationUserSetting.h"

namespace VkRender {
    class Application;

    class ApplicationConfig {
    public:
        struct CRLServerInfo {
            std::string server; // Including prot
            std::string protocol;
            std::string destination;
            std::string versionInfoDestination;
        };

        static ApplicationConfig &getInstance() {
            static ApplicationConfig instance;
            return instance;
        }

        [[nodiscard]] const std::string &getArchitecture() const;

        [[nodiscard]] const std::string &getOsVersion() const;

        [[nodiscard]] const std::string &getAppVersion() const;

        [[nodiscard]] const std::string &getTimeStamp() const;

        [[nodiscard]] const std::string &getOS() const;

        [[nodiscard]] const std::string &getGpuDevice() const;

        [[nodiscard]] const std::string &getAnonymousIdentifier() const;

        [[nodiscard]] const std::vector<std::string> &getEnabledExtensions() const;

        [[nodiscard]] bool hasEnabledExtension(const std::string &extensionName) const;

        void addEnabledExtension(const std::string &extensionName);

        void setGpuDevice(const VkPhysicalDevice &physicalDevice);

        [[nodiscard]]const std::string &getAnonIdentifierString() const;

        [[nodiscard]]const CRLServerInfo &getServerInfo() const {
            return m_ServerInfo;
        }

        [[nodiscard]] const Log::LogLevel &getLogLevel() const {
            return m_UserSetting.logLevel;
        }

        /**
         * Function to set the user settings in the application
         * Example how to set the log level from setUserSetting
         * VkRender::RendererConfig& config = VkRender::RendererConfig::getInstance();
         * auto user = config.getUserSetting();
         * user.logLevel = level;
         * config.setUserSetting(user);
         * Line below also notify the usage monitor which is often combined when user modifies application settings
         * usageMonitor.setSetting("log_level", items[n]);
         * @param setting
         */
        void setUserSetting(const AppConfig::ApplicationUserSetting &setting) {
            m_UserSetting = setting;
            Log::Logger::getInstance()->setLogLevel(setting.logLevel);

        }

        AppConfig::ApplicationUserSetting &getUserSetting();

        AppConfig::ApplicationUserSetting *getUserSettingRef();


        static void removeRuntimeSettingsFile() {
            // If the file exists, delete it
            if (std::filesystem::exists(Utils::getRuntimeConfigFilePath())) {
                std::filesystem::remove(Utils::getRuntimeConfigFilePath());
                Log::Logger::getInstance()->info("Deleted corrupted application settings file at {}",
                                                 Utils::getRuntimeConfigFilePath().string().c_str());
            }
        }

    private:
        ApplicationConfig();



        ~ApplicationConfig() = default;

        CRLServerInfo m_ServerInfo{};
        AppConfig::ApplicationUserSetting m_UserSetting{};
        std::string m_Architecture;
        std::string m_OSVersion;
        std::string m_OS;
        std::string m_AppVersion;
        std::string m_AppVersionRemote;
        std::string m_TimeStamp;
        std::string m_GPUDevice;
        std::string m_Identifier;

        /**@brief holds which extensions are available in this renderer */
        std::vector<std::string> m_EnabledExtensions;


    public:
        const std::string &getAppVersionRemote() const;

        void setAppVersionRemote(const std::string &mAppVersionRemote);
        void saveSettings(Application *ctx);

    private:

        void getOSVersion();

        std::string fetchArchitecture();

        std::string fetchApplicationVersion();

    };

}
#endif //MULTISENSE_VIEWER_RENDERCONFIG_H
