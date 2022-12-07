//
// Created by magnus on 11/29/22.
//

#ifndef MULTISENSE_VIEWER_ADAPTERUTILS_H
#define MULTISENSE_VIEWER_ADAPTERUTILS_H

#include <utility>

#ifdef __linux__
#include <net/if.h>
#include <netinet/in.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif
#ifdef WIN32

#include <iphlpapi.h>
#include <pcap.h>
#endif

namespace Utils {
    struct Adapter {
        std::string ifName;
        uint32_t ifIndex = 0;
        std::string description;
        bool supports = true;
        Adapter(std::string name, uint32_t index) : ifName(std::move(name)), ifIndex(index) {
        }
        Adapter() = default;

    };
#ifdef WIN32

    std::vector<Adapter> listAdapters() {
        std::vector<Adapter> adapters;

        pcap_if_t *alldevs;
        pcap_if_t *d;
        char errbuf[PCAP_ERRBUF_SIZE];

        // Retrieve the device list
        if (pcap_findalldevs(&alldevs, errbuf) == -1) {
            return adapters;
        }

        // Print the list
        int i = 0;
        std::string prefix = "\\Device\\Tcpip_";
        const char *token = "{";
        // Find { token in order to find correct prefix
        for (d = alldevs; d; d = d->next) {
            Adapter adapter(d->name, 0);
            if (d->description)
                adapter.description = d->description;
            size_t found = std::string(d->name).find(token);
            if (found != std::string::npos)
                prefix = std::string(d->name).substr(0, found);
        }

        DWORD dwBufLen = sizeof(IP_ADAPTER_INFO);
        PIP_ADAPTER_INFO AdapterInfo;
        AdapterInfo = (IP_ADAPTER_INFO *) malloc(sizeof(IP_ADAPTER_INFO));
        // Make an initial call to GetAdaptersInfo to get the necessary size into the dwBufLen variable
        if (GetAdaptersInfo(AdapterInfo, &dwBufLen) == ERROR_BUFFER_OVERFLOW) {
            free(AdapterInfo);
            AdapterInfo = (IP_ADAPTER_INFO *) malloc(dwBufLen);
            if (AdapterInfo == NULL) {
                free(AdapterInfo);
                return adapters;
            }
        }
        if (GetAdaptersInfo(AdapterInfo, &dwBufLen) == NO_ERROR) {
            // Contains pointer to current adapter info
            PIP_ADAPTER_INFO pAdapterInfo = AdapterInfo;
            do {
                // Somehow The integrated bluetooth adapter is considered an ethernet adapter cause of the same type in PIP_ADAPTER_INFO field "MIB_IF_TYPE_ETHERNET"
                // I'll filter it out here assuming it has Bluetooth in its name. Just a soft error which increases running time of the auto connect feature
                char *bleFoundInName = strstr(pAdapterInfo->Description, "Bluetooth");
                if (bleFoundInName || pAdapterInfo->Type != MIB_IF_TYPE_ETHERNET) {
                    pAdapterInfo = pAdapterInfo->Next;
                    continue;
                }

                // Internal loopback device always located at index = 1. Skip it..
                if (pAdapterInfo->Index == 1) {
                    pAdapterInfo = pAdapterInfo->Next;
                    continue;
                }

                Adapter adapter("Unnamed", 0);
                adapter.supports = (pAdapterInfo->Type == MIB_IF_TYPE_ETHERNET);
                adapter.description = pAdapterInfo->Description;

                //CONCATENATE two strings safely windows workaround
                int lenA = strlen(prefix.c_str());
                int lenB = strlen(pAdapterInfo->AdapterName);
                char *con = (char *) malloc(lenA + lenB + 1);
                memcpy(con, prefix.c_str(), lenA);
                memcpy(con + lenA, pAdapterInfo->AdapterName, lenB + 1);
                adapter.ifName = con;
                //adapter.networkAdapter = pAdapterInfo->AdapterName;
                adapter.ifIndex = pAdapterInfo->Index;
                adapters.push_back(adapter);
                free(con);
                pAdapterInfo = pAdapterInfo->Next;
            } while (pAdapterInfo);
        }
        free(AdapterInfo);

        return adapters;
    }

#else



    std::vector<Adapter> listAdapters() {
        // Get list of interfaces
        std::vector<Adapter> adapters;
        auto ifn = if_nameindex();
        // If no interfaces. This turns to null if there is no interfaces for a few seconds
        if (!ifn) {
            Log::Logger::getInstance()->error("Failed to list adapters: if_nameindex(), strerror: %s", strerror(errno));
        }
        auto fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
        for (auto i = ifn; i->if_name; ++i) {
            struct {
                struct ethtool_link_settings req{};
                __u32 link_mode_data[3 * 127]{};
            } ecmd{};
            Adapter adapter(i->if_name, i->if_index);

            auto ifr = ifreq{};
            std::strncpy(ifr.ifr_name, i->if_name, IF_NAMESIZE);

            ecmd.req.cmd = ETHTOOL_GLINKSETTINGS;
            ifr.ifr_data = reinterpret_cast<char *>(&ecmd);

            // Check if interface is of type ethernet
            if (ioctl(fd, SIOCETHTOOL, &ifr) == -1) {
                adapter.supports = false;
            }
            // More ethernet checking
            if (ecmd.req.link_mode_masks_nwords >= 0 || ecmd.req.cmd != ETHTOOL_GLINKSETTINGS) {
                adapter.supports = false;
            }
            // Even more ethernet checking
            ecmd.req.link_mode_masks_nwords = -ecmd.req.link_mode_masks_nwords;
            if (ioctl(fd, SIOCETHTOOL, &ifr) == -1) {
                adapter.supports = false;
            }

            if (adapter.supports)
                adapters.emplace_back(adapter);
        }
        if_freenameindex(ifn);
        close(fd);

        return adapters;
    }


#endif // Linux/Win32
};
#endif //MULTISENSE_VIEWER_ADAPTERUTILS_H
