//
// Created by magnus on 4/9/23.
//

#ifndef MULTISENSE_VIEWER_RENDERCONFIG_H
#define MULTISENSE_VIEWER_RENDERCONFIG_H

#include <string>
#include <fstream>
#include <vulkan/vulkan.hpp>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN

#include <windows.h>

#include <winsock2.h>
#include <iphlpapi.h>
#include <cstdlib>

#pragma comment(lib, "IPHLPAPI.lib")
#else
#include <sys/utsname.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include <netinet/in.h>
#include <string.h>

#endif

#include "Viewer/Tools/Logger.h"

namespace VkRender {
    class RendererConfig {
    public:
        struct CRLServerInfo {
            std::string server; // Including prot
            std::string protocol;
            std::string destination;
        };

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

        void setGpuDevice(const VkPhysicalDevice &physicalDevice);

        const std::string &getAnonIdentifierString() const;

        const CRLServerInfo &getServerInfo() const {
            return m_ServerInfo;
        }

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
                                for (int j = 0; j < pIfTable->table[i].dwPhysAddrLen; j++) {
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

            struct ifreq* it = ifc.ifc_req;
            const struct ifreq* const end = it + (ifc.ifc_len / sizeof(struct ifreq));

            for (; it != end; ++it) {
                strcpy(ifr.ifr_name, it->ifr_name);
                if (ioctl(sock, SIOCGIFFLAGS, &ifr) == 0) {
                    if (! (ifr.ifr_flags & IFF_LOOPBACK)) { // don't count loopback
                        if (ioctl(sock, SIOCGIFHWADDR, &ifr) == 0) {
                            success = 1;
                            break;
                        }
                    }
                }
                else { /* handle error */ }
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
        }


        CRLServerInfo m_ServerInfo;

        std::string m_Architecture;
        std::string m_OSVersion;
        std::string m_OS;
        std::string m_AppVersion;
        std::string m_TimeStamp;
        std::string m_GPUDevice;

        std::string m_Identifier;

        void getOSVersion();

        std::string fetchArchitecture();

        std::string fetchApplicationVersion();
    };

};
#endif //MULTISENSE_VIEWER_RENDERCONFIG_H
