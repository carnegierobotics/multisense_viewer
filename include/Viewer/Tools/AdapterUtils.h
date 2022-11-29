//
// Created by magnus on 11/29/22.
//

#ifndef MULTISENSE_VIEWER_ADAPTERUTILS_H
#define MULTISENSE_VIEWER_ADAPTERUTILS_H

#ifdef __linux__
#include <net/if.h>
#include <netinet/in.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <sys/ioctl.h>
#endif
#ifdef WIN32

#endif



#include <utility>
namespace Utils {
    struct Adapter {
        std::string ifName;
        uint32_t ifIndex = 0;
        bool supports = true;
        Adapter(std::string name, uint32_t index) : ifName(std::move(name)), ifIndex(index) {
        }
        Adapter() = default;

    };
#ifdef WIN32

    std::vector<Adapter> listAdapters() {

        std::vector<Adapter> adapters;

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
