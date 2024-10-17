//
// Created by magnus on 4/9/23.
//

#include <random>

#include "ApplicationConfig.h"
#include "Application.h"
#include "ApplicationUserSettingsSerializer.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender {
    ApplicationConfig::ApplicationConfig() {
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

        ApplicationUserSettingsSerializer serializer(m_applicationUserSettings);
        serializer.deserialize(Utils::getRuntimeConfigFilePath());
    }

#ifdef WIN32
    void ApplicationConfig::getOSVersion() {
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

    std::string ApplicationConfig::fetchArchitecture() {
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

    void ApplicationConfig::getOSVersion() {
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

    std::string ApplicationConfig::fetchArchitecture() {
        struct utsname unameData{};
        if (uname(&unameData) != 0) {
            Log::Logger::getInstance()->error("Error calling uname()", unameData.machine);
            return "Unable to read";
        }
        Log::Logger::getInstance()->info("Found architecture: {}", unameData.machine);
        return unameData.machine;
    }
#endif

    std::string ApplicationConfig::fetchApplicationVersion() {
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
                        std::remove(m_ServerInfo.versionInfoDestination.begin(),
                                    m_ServerInfo.versionInfoDestination.end(), '"'),
                        m_ServerInfo.versionInfoDestination.end());
            }
            if (line.find("LOG_LEVEL=") != std::string::npos) {
                std::string logLevelStr = line.substr(line.find("=") + 1);
                logLevelStr.erase(
                        std::remove(logLevelStr.begin(), logLevelStr.end(), '"'),
                        logLevelStr.end());
                m_applicationUserSettings.logLevel = Log::logLevelFromString(logLevelStr);
            }

        }
        Log::Logger::getInstance()->info("Found Application version: {}", applicationVersion);
        m_AppVersionRemote = applicationVersion;
        return applicationVersion;
    }

    const std::string &ApplicationConfig::getArchitecture() const {
        return m_Architecture;
    }

    const std::string &ApplicationConfig::getOsVersion() const {
        return m_OSVersion;
    }

    const std::string &ApplicationConfig::getAppVersion() const {
        return m_AppVersion;
    }

    const std::string &ApplicationConfig::getTimeStamp() const {
        return m_TimeStamp;
    }

    const std::string &ApplicationConfig::getOS() const {
        return m_OS;
    }

    const std::string &ApplicationConfig::getGpuDevice() const {
        return m_GPUDevice;
    }

    void ApplicationConfig::setGpuDevice(VkPhysicalDevice const &physicalDevice) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        m_GPUDevice = properties.deviceName;
    }

    AppConfig::ApplicationUserSettings &ApplicationConfig::getUserSetting() {
        return m_applicationUserSettings;
    }

    AppConfig::ApplicationUserSettings *ApplicationConfig::getUserSettingRef() {
        return &m_applicationUserSettings;
    }

    const std::string &ApplicationConfig::getAnonymousIdentifier() const {
        return m_Identifier;
    }

    const std::string &ApplicationConfig::getAppVersionRemote() const {
        return m_AppVersionRemote;
    }

    void ApplicationConfig::setAppVersionRemote(const std::string &mAppVersionRemote) {
        m_AppVersionRemote = mAppVersionRemote;
    }

    const std::vector<std::string> &ApplicationConfig::getEnabledExtensions() const {
        return m_EnabledExtensions;
    }

    void ApplicationConfig::addEnabledExtension(const std::string &extensionName) {
        m_EnabledExtensions.emplace_back(extensionName);
    }

    bool ApplicationConfig::hasEnabledExtension(const std::string &extensionName) const {
        return Utils::isInVector(m_EnabledExtensions, extensionName);
    }

    void ApplicationConfig::saveSettings() {
        // Save application settings to file
        ApplicationUserSettingsSerializer serializer(m_applicationUserSettings);
        serializer.serialize(Utils::getRuntimeConfigFilePath().string());
        Log::Logger::getInstance()->info("Saved editor runtime settings to: {}",
                                         Utils::getRuntimeConfigFilePath().string());

    }


}
