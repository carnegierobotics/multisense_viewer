//
// Created by magnus on 4/9/23.
//

#ifndef MULTISENSE_VIEWER_RENDERCONFIG_H
#define MULTISENSE_VIEWER_RENDERCONFIG_H

#include <string>
#include <fstream>
#include <vulkan/vulkan.hpp>
#include <random>

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
#include "Viewer/Renderer/ApplicationUserSetting.h"

namespace VkRender {
    class RendererConfig {
    public:
        struct CRLServerInfo {
            std::string server; // Including prot
            std::string protocol;
            std::string destination;
            std::string versionInfoDestination;
        };

        static RendererConfig &getInstance() {
            static RendererConfig instance;
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

    private:
        RendererConfig() {
            getOSVersion();
            m_Architecture = fetchArchitecture();
            m_AppVersion = fetchApplicationVersion();


#ifdef WIN32
            std::ostringstream macAddressStream;
            DWORD dwSize = 0;
            DWORD dwRetVal = 0;

            MIB_IFTABLE *pIfTable;
            dwRetVal = GetIfTable(nullptr, &dwSize, false);
            if (dwRetVal == ERROR_INSUFFICIENT_BUFFER) {
                pIfTable = reinterpret_cast<MIB_IFTABLE *>(malloc(dwSize));
                if (pIfTable == nullptr) {
                    Log::Logger::getInstance()->error("Error allocating memory for getting mac address");
                } else {
                    dwRetVal = GetIfTable(pIfTable, &dwSize, false);
                    if (dwRetVal == NO_ERROR) {
                        for (DWORD i = 0; i < pIfTable->dwNumEntries; i++) {
                            if (pIfTable->table[i].dwType == IF_TYPE_ETHERNET_CSMACD) {
                                for (DWORD j = 0; j < pIfTable->table[i].dwPhysAddrLen; j++) {
                                    macAddressStream << std::hex << std::setw(2) << std::setfill('0')
                                                     << static_cast<int>(pIfTable->table[i].bPhysAddr[j]);
                                    if (j < pIfTable->table[i].dwPhysAddrLen - 1) {
                                        macAddressStream << ':';
                                    }
                                }
                                macAddressStream << std::endl;
                                break;
                            }
                        }
                    } else {
                        Log::Logger::getInstance()->error("Error getting interface table: {}", dwRetVal);
                    }
                    free(pIfTable);
                }
            } else {
                Log::Logger::getInstance()->error("Error getting interface table size: {}", dwRetVal);
            }
            std::string macAddress = macAddressStream.str();
#else
            struct ifreq ifr;
            struct ifconf ifc;
            char buf[1024];
            int success = 0;

            int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
            if (sock == -1) { /* handle error*/ };

            ifc.ifc_len = sizeof(buf);
            ifc.ifc_buf = buf;
            if (ioctl(sock, SIOCGIFCONF, &ifc) == -1) { /* handle error */ }

            struct ifreq *it = ifc.ifc_req;
            const struct ifreq *const end = it + (ifc.ifc_len / sizeof(struct ifreq));

            for (; it != end; ++it) {
                strcpy(ifr.ifr_name, it->ifr_name);
                if (ioctl(sock, SIOCGIFFLAGS, &ifr) == 0) {
                    if (!(ifr.ifr_flags & IFF_LOOPBACK)) { // don't count loopback
                        if (ioctl(sock, SIOCGIFHWADDR, &ifr) == 0) {
                            success = 1;
                            break;
                        }
                    }
                } else { /* handle error */ }
            }

            std::string macAddress;
            macAddress.resize(14);

            if (success) memcpy(macAddress.data(), ifr.ifr_hwaddr.sa_data, 14);
#endif

            // Seed the random number generator with the MAC address
            std::seed_seq seed{std::hash<std::string>{}(macAddress)};
            std::mt19937 gen(seed);
            // Generate a random string
            std::uniform_int_distribution<int> dist('a', 'z');
            std::string random_string;
            for (int i = 0; i < 10; ++i) {
                random_string += static_cast<char>(dist(gen));
            }
            m_Identifier = random_string;
            Log::Logger::getInstance()->info("Generated Random Identifier: {}", m_Identifier);

            // LOAD APPLICATION SETTINGS
            // Deserialize settings
            // Read JSON from file
            std::filesystem::path settingsFilePath = Utils::getSystemCachePath() / "AppRuntimeConfig.json";
            try {

                if (std::filesystem::exists(settingsFilePath)) {

                    std::ifstream inFile(settingsFilePath);
                    nlohmann::json j_in;
                    inFile >> j_in;
                    inFile.close();
                    m_UserSetting = j_in.template get<AppConfig::ApplicationUserSetting>();
                    Log::Logger::getInstance()->info("Loaded application settings from {}",
                                                     settingsFilePath.string().c_str());
                } else {
                    Log::Logger::getInstance()->info("No application settings file at {}",
                                                     settingsFilePath.string().c_str());
                }
            } catch (const std::exception& e) {
                Log::Logger::getInstance()->error("Failed to load application settings: {}", e.what());
                // If the file exists, delete it
                if (std::filesystem::exists(settingsFilePath)) {
                    std::filesystem::remove(settingsFilePath);
                    Log::Logger::getInstance()->info("Deleted corrupted application settings file at {}",
                                                     settingsFilePath.string().c_str());
                }
            }

        }
        ~RendererConfig() {
            // Save application settings to file
            // Save application settings to file
            try {
                // Serialize settings
                nlohmann::json j_out = m_UserSetting;

                // Path where settings will be saved
                std::filesystem::path settingsFilePath = Utils::getSystemCachePath() / "AppRuntimeConfig.json";

                // Write JSON to file
                std::ofstream outFile(settingsFilePath);
                if (outFile.is_open()) {
                    outFile << std::setw(4) << j_out << std::endl;
                    outFile.close();
                    Log::Logger::getInstance()->info("Saved application settings to {}", settingsFilePath.string().c_str());
                } else {
                    Log::Logger::getInstance()->info("Failed to open settings file at {}", settingsFilePath.string().c_str());
                }
            } catch (const std::exception& e) {
                Log::Logger::getInstance()->error("Exception while saving settings: {}", e.what());
            }
        }

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

    private:

        void getOSVersion();

        std::string fetchArchitecture();

        std::string fetchApplicationVersion();
    };

}
#endif //MULTISENSE_VIEWER_RENDERCONFIG_H
