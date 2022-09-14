//
// Created by magnus on 7/14/22.
//

#include <iostream>
#include <cstring>
#include <linux/sockios.h>
#include <net/if.h>
#include <execution>
#include <linux/ethtool.h>
#include <netinet/ip.h>
#include <sys/ioctl.h>
#include <MultiSense/details/channel.hh>
#include <arpa/inet.h>
#include "AutoConnectLinux.h"
#include <netpacket/packet.h>
#include <sys/fcntl.h>

std::vector<AutoConnect::Result> AutoConnectLinux::findEthernetAdapters(bool logEvent, bool skipIgnored) {
    std::vector<Result> adapterSupportResult;
    auto ifn = if_nameindex();
    auto fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    for (auto i = ifn; i->if_name; ++i) {
        struct {
            struct ethtool_link_settings req;
            __u32 link_mode_data[3 * 127];
        } ecmd{};
        Result adapter(i->if_name, false);


        // Filter out docker adapters.
        if (strstr(i->if_name, "docker") != NULL)
            continue;


        auto ifr = ifreq{};
        std::strncpy(ifr.ifr_name, i->if_name, IF_NAMESIZE);

        ecmd.req.cmd = ETHTOOL_GLINKSETTINGS;
        ifr.ifr_data = reinterpret_cast<char *>(&ecmd);

        if (ioctl(fd, SIOCETHTOOL, &ifr) == -1) {
            continue;

        }
        if (ecmd.req.link_mode_masks_nwords >= 0 || ecmd.req.cmd != ETHTOOL_GLINKSETTINGS) {
            continue;
        }
        ecmd.req.link_mode_masks_nwords = -ecmd.req.link_mode_masks_nwords;
        if (ioctl(fd, SIOCETHTOOL, &ifr) == -1) {
            continue;
        }

        adapter.supports = true;
        adapter.description = adapter.networkAdapter; // Copy name to description
        adapter.index = i->if_index;

        // If a camera has al ready been found on the adapter then ignore it. Remove it from adapters list
        bool ignore = false;
        for (const auto &found: ignoreAdapters) {
            if (found.index == adapter.index) {
                ignore = true;
                std::string str = "Found already searched adapter: " + adapter.networkAdapter + ". Ignoring...";
                eventCallback(str, context, 0);
            }
        }

        if (ignore && !skipIgnored) {
            adapter.searched = true;
        }

        adapterSupportResult.emplace_back(adapter);
    }


    if (!adapterSupportResult.empty())
        onFoundAdapters(adapterSupportResult, logEvent);
    return adapterSupportResult;
}

void AutoConnectLinux::run(void *instance, std::vector<Result> adapters) {
    AutoConnectLinux *app = (AutoConnectLinux *) instance;
    app->eventCallback("Started detection service", app->context, 0);
    // Get list of network adapters that are  supports our application
    std::string hostAddress;
    int i = 0;
    // Loop keeps retrying to connect on supported network adapters.
    while (app->loopAdapters) {
        if (i >= adapters.size()) {
            i = 0;
            app->eventCallback("Tested all adapters. rerunning adapter search", app->context, 0);
            adapters = app->findEthernetAdapters(true, false);
            bool testedAllAdapters = true;
            for (const auto& a : adapters){
                if (a.supports && !a.searched)
                    testedAllAdapters = false;
            }
            if (testedAllAdapters) {
                app->eventCallback("No other adapters found", app->context, 0);
                app->eventCallback("Finished", app->context, 0);
                break;
            }
        }
        auto adapter = adapters[i];
        // If it doesn't support a camera then dont loop it
        if (!adapter.supports) {
            continue;
        }
        // If a camera has al ready been found on the adapter then dont re-run a search on it. Remove it from adapters list
        bool isAlreadySearched = false;
        for (const auto &found: app->ignoreAdapters) {
            if (found.index == adapter.index)
                isAlreadySearched = true;
        }
        if (isAlreadySearched) {
            adapters.erase(adapters.begin() + i);
            continue;
        }
        i++;
        std::string str = "Testing Adapter. Name: " + adapter.networkAdapter;
        app->eventCallback(str, app->context, 0);
        app->startTime = time(nullptr);
        int sd = -1;
        // Submit request for a socket descriptor to look up interface.
        if ((sd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_IP))) < 0) {
            perror("socket() failed to get socket descriptor for using ioctl() ");
            app->eventCallback("Error", app->context, 2);
            continue;
        }
        std::cout << "Created socket: " << sd << ". Binding to device: " << adapter.networkAdapter << std::endl;
        struct sockaddr_ll  addr;
        // Bind socket to interface
        memset(&addr, 0x00, sizeof(addr));
        addr.sll_family   = AF_PACKET;
        addr.sll_protocol = htons(ETH_P_ALL);
        addr.sll_ifindex  = adapter.index;
        if (bind(sd, (struct sockaddr*)&addr, sizeof(addr)) == -1){
            std::cout << "ERROR IN BIND: " << strerror(errno) << std::endl;
            perror("Bind");
        }
        /* set the network card in promiscuos mode*/
        // An ioctl() request has encoded in it whether the argument is an in parameter or out parameter
        // SIOCGIFFLAGS	0x8913		/* get flags			*/
        // SIOCSIFFLAGS	0x8914		/* set flags			*/
        struct ifreq ethreq;
        strncpy(ethreq.ifr_name, adapter.networkAdapter.c_str(), IF_NAMESIZE);
        if (ioctl(sd, SIOCGIFFLAGS, &ethreq) == -1) {
            std::cout << "Error: " << adapter.networkAdapter << "socket: " << sd << std::endl;
            perror("SIOCGIFFLAGS: ioctl");
            app->eventCallback(std::string(std::string("Error: ") + adapter.networkAdapter), app->context, 2);
            continue;
        }
        ethreq.ifr_flags |= IFF_PROMISC;

        if (ioctl(sd, SIOCSIFFLAGS, &ethreq) == -1) {
            perror("SIOCSIFFLAGS: ioctl");
            app->eventCallback(std::string(std::string("Error: ") + adapter.networkAdapter), app->context, 2);
            continue;
        }

        str = "Set adapter to listen for activity";
        app->eventCallback(str, app->context, 0);

        int saddr_size, data_size;
        struct sockaddr saddr{};
        auto *buffer = (unsigned char *) malloc(IP_MAXPACKET + 1);

        str = "Waiting for packet at: " + adapter.networkAdapter;
        app->eventCallback(str, app->context, 0);
        while (app->listenOnAdapter) {
            // Timeout handler
            if ((time(nullptr) - app->startTime) > TIMEOUT_INTERVAL_SECONDS) {
                app->startTime = time(nullptr);
                printf("\n");
                str = "Timeout reached. switching to next supported adapter ";
                app->eventCallback(str, app->context, 2);
                break;
            }

            saddr_size = sizeof saddr;
            //Receive a packet
            data_size = (int) recvfrom(sd, buffer, IP_MAXPACKET, MSG_DONTWAIT, &saddr,
                                       (socklen_t *) &saddr_size);

            if (data_size < 0) {
                //printf("Recvfrom error , failed to get packets %s\n", strerror(errno));
                continue;
            }

            //Now process the packet
            auto *iph = (struct iphdr *) (buffer + sizeof(struct ethhdr));
            struct in_addr ip_addr{};
            std::string address;
            if (iph->protocol == IPPROTO_IGMP) //Check the Protocol and do accordingly...
            {
                ip_addr.s_addr = iph->saddr;
                address = inet_ntoa(ip_addr);
                str = "Packet found. Source address: " + address;
                app->eventCallback(str, app->context, 0);

                FoundCameraOnIp ret = app->onFoundIp(address, adapter, sd);

                if (ret == FOUND_CAMERA) {
                    app->ignoreAdapters.push_back(adapter);
                    app->onFoundCamera(adapter);
                    close(sd);
                    break;
                } else if (ret == NO_CAMERA_RETRY) {
                    continue;
                } else if (ret == NO_CAMERA) {
                    close(sd);
                    break;
                }

            }
        }
    }

    printf("Exited thread\n");
}

void AutoConnectLinux::onFoundAdapters(std::vector<Result> adapters, bool logEvent) {

    for (auto &adapter: adapters) {
        if (adapter.supports && !adapter.searched) {
            std::string str;
            str = "Found supported adapter: " + adapter.networkAdapter;
            if (logEvent)
                eventCallback(str, context, 1);
        }
    }

}

AutoConnect::FoundCameraOnIp AutoConnectLinux::onFoundIp(std::string address, Result adapter, int camera_fd) {
// Set the host ip address to the same subnet but with *.1 at the end.
    std::string hostAddress = address;
    std::string last_element(hostAddress.substr(hostAddress.rfind(".")));
    auto ptr = hostAddress.rfind('.');
    hostAddress.replace(ptr, last_element.length(), ".2");

    std::string str = "Setting host address to: " + hostAddress;
    eventCallback(str, context, 0);

    /*** CALL IOCTL Operations to set the address of the adapter/socket  ***/

    struct ifreq ifr{};
    /// note: no pointer here
    struct sockaddr_in inet_addr{}, subnet_mask{};
    /* get interface name */
    /* Prepare the struct ifreq */
    bzero(ifr.ifr_name, IFNAMSIZ);
    strncpy(ifr.ifr_name, adapter.networkAdapter.c_str(), IFNAMSIZ);

    /// note: prepare the two struct sockaddr_in
    inet_addr.sin_family = AF_INET;
    int inet_addr_config_result = inet_pton(AF_INET, hostAddress.c_str(), &(inet_addr.sin_addr));

    subnet_mask.sin_family = AF_INET;
    int subnet_mask_config_result = inet_pton(AF_INET, "255.255.255.0", &(subnet_mask.sin_addr));


    // Call ioctl to configure network devices
    /// put addr in ifr structure
    memcpy(&(ifr.ifr_addr), &inet_addr, sizeof(struct sockaddr));
    int ioctl_result = ioctl(camera_fd, SIOCSIFADDR, &ifr);  // Set IP address
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFADDR: %s\n", strerror(errno));
        eventCallback("Error", context, 0);
        return NO_CAMERA;
    }

    /// put mask in ifr structure
    memcpy(&(ifr.ifr_addr), &subnet_mask, sizeof(struct sockaddr));
    ioctl_result = ioctl(camera_fd, SIOCSIFNETMASK, &ifr);   // Set subnet mask
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFNETMASK: ");
        eventCallback("Error", context, 0);
        return NO_CAMERA;
    }
    /*** END **/

    // Attempt to connect to camera and post some info
    str = "Checking for camera at: " + address;
    eventCallback(str, context, 0);

    std::cout << "Camera interface: " << adapter.index << " name: " << adapter.networkAdapter << std::endl;
    cameraInterface = crl::multisense::Channel::Create(address);

    if (cameraInterface == nullptr && connectAttemptCounter > MAX_CONNECTION_ATTEMPTS) {
        connectAttemptCounter = 0;
        // Attempt to connect to camera and post some info
        str = "Did not detect camera at: " + address + ". Switching adapter...";
        eventCallback(str, context, 2);
        return NO_CAMERA;
    } else if (cameraInterface == nullptr) {
        connectAttemptCounter++;
        str = "Did not detect camera at: " + address + ". Retrying...";
        eventCallback(str, context, 2);
        return NO_CAMERA_RETRY;
    } else {
        result.networkAdapter = adapter.networkAdapter;
        result.networkAdapterLongName = adapter.networkAdapterLongName;
        result.cameraIpv4Address = address;
        result.index = adapter.index;
        str = "Found camera at: " + address + "";
        eventCallback(str, context, 1);
        return FOUND_CAMERA;
    }
}

void AutoConnectLinux::onFoundCamera(Result supportResult) {
    success = true;
    callback(result, context);

}

void AutoConnectLinux::stop() {
    loopAdapters = false;
    listenOnAdapter = false;
    shouldProgramRun = false;
    running = false;

    if (t != nullptr)
        t->join();


    delete (t); //instantiated in start func
    t = nullptr;

}

void AutoConnectLinux::start(std::vector<Result> adapters) {

    // TODO Clean up public booleans. 4 member booleans might be exaggerated use?
    running = true;
    loopAdapters = true;
    listenOnAdapter = true;
    shouldProgramRun = true;

    t = new std::thread(&AutoConnectLinux::run, this, adapters);
}

AutoConnect::Result AutoConnectLinux::getResult() {
    return result;
}

crl::multisense::Channel *AutoConnectLinux::getCameraChannel() {
    return cameraInterface;
}

void AutoConnectLinux::setDetectedCallback(void (*param)(Result result1, void *ctx), void *context) {
    callback = param;
    this->context = context;

}

bool AutoConnectLinux::shouldProgramClose() {
    return !shouldProgramRun; // Note: This is just confusing usage... future technical debt right here
}

void AutoConnectLinux::setShouldProgramClose(bool close) {
    this->shouldProgramRun = !close; // Note: This is just confusing usage... future technical debt right here
}

void AutoConnectLinux::setEventCallback(void (*param)(std::string str, void *, int)) {
    eventCallback = param;

}

void AutoConnectLinux::clearSearchedAdapters() {
    ignoreAdapters.clear();
}
