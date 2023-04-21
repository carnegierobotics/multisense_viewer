//
// Created by magnus on 4/9/23.
//

#ifndef MULTISENSE_VIEWER_RENDERCONFIG_H
#define MULTISENSE_VIEWER_RENDERCONFIG_H

#include <string>
#include <fstream>
#include <vulkan/vulkan.hpp>

#ifdef WIN32
#include <iphlpapi.h>
#include <WinSock2.h>
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

        void setGpuDevice(const VkPhysicalDevice& physicalDevice);

        const std::string & getAnonIdentifierString() const;

        const CRLServerInfo & getServerInfo() const {
            return m_ServerInfo;
        }

    private:
        RendererConfig() {
            getOSVersion();
            m_Architecture = fetchArchitecture();
            m_AppVersion = fetchApplicationVersion();


#ifdef WIN32
            PIP_ADAPTER_INFO adapter_info;
    PIP_ADAPTER_INFO adapter;
    DWORD size = 0;

    GetAdaptersInfo(nullptr, &size);
    adapter_info = (IP_ADAPTER_INFO*) malloc(size);
    GetAdaptersInfo(adapter_info, &size);

                std::string macAddress;
            macAddress.resize(14);

    adapter = adapter_info;
    while (adapter != nullptr) {
        printf("MAC address for interface %s: %02x:%02x:%02x:%02x:%02x:%02x\n",
               adapter->AdapterName,
               adapter->Address[0], adapter->Address[1], adapter->Address[2],
               adapter->Address[3], adapter->Address[4], adapter->Address[5]);
        adapter = adapter->Next;
    }

    free(adapter_info);
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
