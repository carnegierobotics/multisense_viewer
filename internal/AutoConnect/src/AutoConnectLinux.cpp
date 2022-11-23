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
#include <arpa/inet.h>
#include <netpacket/packet.h>
#include <mutex>

#include "AutoConnect/AutoConnectLinux.h"


void AutoConnectLinux::run(){
    while (true){
        {
            std::scoped_lock<std::mutex> lock(m_AdaptersMutex);
            for (auto& item: m_Adapters){
                if (item.supports && item.available) {
                    item.available = false;
                    m_Pool->Push(AutoConnectLinux::listenOnAdapter, this, &item);
                }
            }
        }

        {
            std::scoped_lock<std::mutex> lock(m_AdaptersMutex);
            for (auto& item: m_Adapters){
                m_Pool->Push(AutoConnectLinux::checkForCamera, this, &item);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void AutoConnectLinux::adapterScan(void *ctx) {
    auto *app = static_cast<AutoConnectLinux *>(ctx);

    while (app->m_ScanAdapters) {
        // Get list of interfaces
        std::vector<Adapter> adapters;
        auto ifn = if_nameindex();
        // If no interfaces. This turns to null if there is no interfaces for a few seconds
        if (!ifn) {
            fprintf(stderr, "if_nameindex error: %s\n", strerror(errno));
            continue;
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

            adapters.emplace_back(adapter);
        }
        if_freenameindex(ifn);
        close(fd);

        // Put into shared list
        // If the name is new then insert it in the list
        {
            std::scoped_lock<std::mutex> lock(app->m_AdaptersMutex);
            for (const auto &adapter: adapters){
                bool exist = false;
                for (const auto &shared: app->m_Adapters){
                    if (shared.ifName == adapter.ifName)
                        exist = true;
                }
                if (!exist){
                    app->m_Adapters.emplace_back(adapter);
                    printf("Found adapter %s %d supported = %d\n", adapter.ifName.c_str(), adapter.ifIndex, adapter.supports);
                }
            }
        }
        // Don't update too fast
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void AutoConnectLinux::listenOnAdapter(void *ctx, Adapter *adapter) {
    auto* app = static_cast<AutoConnectLinux*>(ctx);
    // Submit request for a socket descriptor to look up interface.
    int sd = 0;
    if ((sd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_IP))) < 0) {
        perror("socket() failed to get socket descriptor for using ioctl() ");
    }


    std::cout << "Created socket: " << sd << ". Binding to device: " << adapter->ifName << std::endl;

    struct sockaddr_ll addr{};
    // Bind socket to interface
    memset(&addr, 0x00, sizeof(addr));
    addr.sll_family = AF_PACKET;
    addr.sll_protocol = htons(ETH_P_ALL);
    addr.sll_ifindex = (int) adapter->ifIndex;
    if (bind(sd, (struct sockaddr *) &addr, sizeof(addr)) == -1) {
        std::cout << "ERROR IN BIND: " << strerror(errno) << std::endl;
        perror("Bind");
    }
    // set the network card in promiscuos mode//
    // An ioctl() request has encoded in it whether the argument is an in parameter or out parameter
    // SIOCGIFFLAGS	0x8913		// get flags			//
    // SIOCSIFFLAGS	0x8914		// set flags			//
    struct ifreq ethreq{};
    strncpy(ethreq.ifr_name, adapter->ifName.c_str(), IF_NAMESIZE);
    if (ioctl(sd, SIOCGIFFLAGS, &ethreq) == -1) {
        std::cout << "Error: " << adapter->ifName << "socket: " << sd << std::endl;
        perror("SIOCGIFFLAGS: ioctl");
    }
    ethreq.ifr_flags |= IFF_PROMISC;

    if (ioctl(sd, SIOCSIFFLAGS, &ethreq) == -1) {
        perror("SIOCSIFFLAGS: ioctl");}

    int saddr_size, data_size;
    struct sockaddr saddr{};
    auto *buffer = (unsigned char *) malloc(IP_MAXPACKET + 1);
    auto startListenTime = std::chrono::steady_clock::now();
    float timeOut = 10.0f;
    while (true) {
        // Timeout handler
        // Will timeout MAX_CONNECTION_ATTEMPTS times until retrying on new adapter
        auto time_span = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::steady_clock::now() - startListenTime);
        // 10 Seconds, then alert user and break loop
        if (time_span.count() > timeOut){
            break;
        }
        saddr_size = sizeof saddr;
        //Receive a packet
        data_size = (int) recvfrom(sd, buffer, IP_MAXPACKET, MSG_DONTWAIT, &saddr,
                                   (socklen_t *) &saddr_size);

        if (data_size < 0) {
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
            // If not already in vector


            std::cout << "Address: " << address << " On adapter: " << adapter->ifName<<  std::endl;

            std::scoped_lock<std::mutex> lock(app->m_AdaptersMutex);
            if (std::find(adapter->IPAddresses.begin(), adapter->IPAddresses.end(), address) == adapter->IPAddresses.end()){
                adapter->IPAddresses.emplace_back(address);
            }
        }
    }

    std::cout << "Finished search on " << adapter->ifName << std::endl;
    free(buffer);
}

void AutoConnectLinux::checkForCamera(void* ctx, Adapter* adapter){

    std::string address = adapter->IPAddresses.front();
    std::string adapterName = adapter->ifName;
    int cameraFD = -1;
    // Set the host ip address to the same subnet but with *.2 at the end.
    std::string hostAddress = address;
    std::string last_element(hostAddress.substr(hostAddress.rfind(".")));
    auto ptr = hostAddress.rfind('.');
    hostAddress.replace(ptr, last_element.length(), ".2");


    // CALL IOCTL Operations to set the address of the adapter/socket  //
    struct ifreq ifr{};
    /// note: no pointer here
    struct sockaddr_in inet_addr{}, subnet_mask{};
    // get interface name //
    // Prepare the struct ifreq //
    bzero(ifr.ifr_name, IFNAMSIZ);
    strncpy(ifr.ifr_name, adapterName.c_str(), IFNAMSIZ);

    /// note: prepare the two struct sockaddr_in
    inet_addr.sin_family = AF_INET;
    inet_pton(AF_INET, hostAddress.c_str(), &(inet_addr.sin_addr));

    subnet_mask.sin_family = AF_INET;
    inet_pton(AF_INET, "255.255.255.0", &(subnet_mask.sin_addr));


    // Call ioctl to configure network devices
    /// put addr in ifr structure
    memcpy(&(ifr.ifr_addr), &inet_addr, sizeof(struct sockaddr));
    int ioctl_result = ioctl(cameraFD, SIOCSIFADDR, &ifr);  // Set IP address
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFADDR: %s\n", strerror(errno));
    }

    /// put mask in ifr structure
    memcpy(&(ifr.ifr_addr), &subnet_mask, sizeof(struct sockaddr));
    ioctl_result = ioctl(cameraFD, SIOCSIFNETMASK, &ifr);   // Set subnet mask
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFNETMASK: ");
    }

    auto* channelPtr = crl::multisense::Channel::Create(address, adapterName);

    std::cout << "Camera interface: " << adapter.index << " name: " << adapter.networkAdapter << std::endl;

}
/*
void AutoConnectLinux::run(void *ctx) {
    std::vector<Result> adapters{};
    auto *app = (AutoConnectLinux *) ctx;
    app->m_EventCallback("Started detection service", app->m_Context, 0);
    app->shutdownT1Ready = false;
    std::string hostAddress;
    size_t i = 0;
    // Loop keeps retrying to connect on supported network adapters.
    while (app->m_LoopAdapters) {
        if (i >= adapters.size()) {
            i = 0;
            app->m_EventCallback("Running adapter search", app->m_Context, 0);
            adapters.clear();
            app->findEthernetAdapters(ctx, true, false, &adapters);
            if (adapters.empty()) {
                app->m_EventCallback("No adapters found", app->m_Context, 2);
                app->m_EventCallback("Finished", app->m_Context,
                                     0); // This nessage actually sends a stop call to the gui
                break;
            }
            bool testedAllAdapters = !app->m_IgnoreAdapters.empty();
            for (auto &a: adapters) {
                for (const auto &ignore: app->m_IgnoreAdapters) {
                    if (ignore.networkAdapter == a.networkAdapter)
                        a.searched = true;
                }

                if (!a.searched)
                    testedAllAdapters = false;
            }

            if (testedAllAdapters) {
                app->m_EventCallback(adapters.empty() ? "No adapters found" : "No other adapters found", app->m_Context,
                                     0);
                app->m_EventCallback("Finished", app->m_Context, 0);
                break;
            }
        }

        if (app->interrupt) {
            app->shutdownT1Ready = true;
            return;
        }

        Result adapter{};
        adapter = adapters[i];
        // If it doesn't support a camera then dont loop it
        if (!adapter->supports) {
            continue;
        }
        // If a camera has al ready been found on the adapter then dont re-run a search on it. Remove it from adapters list
        bool isAlreadySearched = false;
        for (const auto &found: app->m_IgnoreAdapters) {
            if (found.index == adapter->index)
                isAlreadySearched = true;
        }
        if (isAlreadySearched) {
            adapters.erase(adapters.begin() + i);
            continue;
        }
        i++;
        std::string str = "Testing Adapter. Name: " + adapter->networkAdapter;
        app->m_EventCallback(str, app->m_Context, 0);
        app->startTime = time(nullptr);
        int sd = -1;

        // Submit request for a socket descriptor to look up interface.
        if ((sd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_IP))) < 0) {
            perror("socket() failed to get socket descriptor for using ioctl() ");
            app->m_EventCallback("Error", app->m_Context, 2);
            continue;
        }
        std::cout << "Created socket: " << sd << ". Binding to device: " << adapter.networkAdapter << std::endl;
        struct sockaddr_ll addr{};
        // Bind socket to interface
        memset(&addr, 0x00, sizeof(addr));
        addr.sll_family = AF_PACKET;
        addr.sll_protocol = htons(ETH_P_ALL);
        addr.sll_ifindex = adapter.index;
        if (bind(sd, (struct sockaddr *) &addr, sizeof(addr)) == -1) {
            std::cout << "ERROR IN BIND: " << strerror(errno) << std::endl;
            perror("Bind");
        }
        // set the network card in promiscuos mode//
        // An ioctl() request has encoded in it whether the argument is an in parameter or out parameter
        // SIOCGIFFLAGS	0x8913		// get flags			//
        // SIOCSIFFLAGS	0x8914		// set flags			//
        struct ifreq ethreq{};
        strncpy(ethreq.ifr_name, adapter.networkAdapter.c_str(), IF_NAMESIZE);
        if (ioctl(sd, SIOCGIFFLAGS, &ethreq) == -1) {
            std::cout << "Error: " << adapter.networkAdapter << "socket: " << sd << std::endl;
            perror("SIOCGIFFLAGS: ioctl");
            app->m_EventCallback(std::string(std::string("Error: ") + adapter.networkAdapter), app->m_Context, 2);
            continue;
        }
        ethreq.ifr_flags |= IFF_PROMISC;

        if (ioctl(sd, SIOCSIFFLAGS, &ethreq) == -1) {
            perror("SIOCSIFFLAGS: ioctl");
            app->m_EventCallback(std::string(std::string("Error: ") + adapter.networkAdapter), app->m_Context, 2);
            continue;
        }

        str = "Set adapter to listen for activity";
        app->m_EventCallback(str, app->m_Context, 0);

        if (app->interrupt) {
            app->shutdownT1Ready = true;
            return;
        }
        int saddr_size, data_size;
        struct sockaddr saddr{};
        auto *buffer = (unsigned char *) malloc(IP_MAXPACKET + 1);
        str = "Waiting for packet at: " + adapter.networkAdapter;
        app->m_EventCallback(str, app->m_Context, 0);
        while (app->m_ListenOnAdapter) {
            // Timeout handler
            // Will timeout MAX_CONNECTION_ATTEMPTS times until retrying on new adapter
            if ((time(nullptr) - app->startTime) > TIMEOUT_INTERVAL_SECONDS &&
                app->connectAttemptCounter < MAX_CONNECTION_ATTEMPTS) {
                app->startTime = time(nullptr);
                printf("\n");
                str = "Timeout reached. Retrying... (" + std::to_string(app->connectAttemptCounter + 1) + "/" +
                      std::to_string(MAX_CONNECTION_ATTEMPTS) + ")";
                app->m_EventCallback(str, app->m_Context, 0);
                app->connectAttemptCounter++;
                str = "Waiting for packet at: " + adapter.networkAdapter;
                app->m_EventCallback(str, app->m_Context, 0);
            } else if ((time(nullptr) - app->startTime) > TIMEOUT_INTERVAL_SECONDS &&
                       app->connectAttemptCounter >= MAX_CONNECTION_ATTEMPTS) {
                app->startTime = time(nullptr);
                printf("\n");
                str = "Timeout reached. Switching to next supported adapter";
                app->m_EventCallback(str, app->m_Context, 2);
                app->connectAttemptCounter = 0;
                app->m_IgnoreAdapters.push_back(adapter);
                break;

            }
            if (app->interrupt) {
                app->shutdownT1Ready = true;
                free(buffer);
                return;
            }

            saddr_size = sizeof saddr;
            //Receive a packet
            data_size = (int) recvfrom(sd, buffer, IP_MAXPACKET, MSG_DONTWAIT, &saddr,
                                       (socklen_t *) &saddr_size);

            if (data_size < 0) {
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
                app->m_EventCallback(str, app->m_Context, 0);


                FoundCameraOnIp ret = app->onFoundIp(address, adapter, sd);

                if (app->interrupt) {
                    app->shutdownT1Ready = true;
                    free(buffer);
                    return;
                }
                if (ret == FOUND_CAMERA) {
                    app->m_IgnoreAdapters.push_back(adapter);
                    app->onFoundCamera();
                    // Disable promscious mode
                    ethreq.ifr_flags &= ~IFF_PROMISC;
                    if (ioctl(sd, SIOCSIFFLAGS, &ethreq) == -1) {
                        perror("SIOCSIFFLAGS: ioctl");
                        app->m_EventCallback(std::string(std::string("Error: ") + adapter.networkAdapter),
                                             app->m_Context, 2);
                        continue;
                    }
                    close(sd);
                    sd = -1;
                    break;
                } else if (ret == NO_CAMERA_RETRY) {
                    continue;
                } else if (ret == NO_CAMERA) {
                    close(sd);
                    sd = -1;
                    break;
                }
            }

        }
        free(buffer);
        // If we were interrupted, then reset promiscuous mode and close socket
        if (sd != -1) {
            ethreq.ifr_flags &= ~IFF_PROMISC;
            if (ioctl(sd, SIOCSIFFLAGS, &ethreq) == -1) {
                perror("SIOCSIFFLAGS: ioctl");
                app->m_EventCallback(std::string(std::string("Error: ") + adapter.networkAdapter),
                                     app->m_Context, 2);
                close(sd);
            }
        }
    }

    app->shutdownT1Ready = true;

}

void AutoConnectLinux::onFoundAdapters(std::vector<Result> adapters, bool logEvent) {

}

AutoConnect::FoundCameraOnIp AutoConnectLinux::onFoundIp(std::string address, Result adapter, int camera_fd) {
    // Set the host ip address to the same subnet but with *.2 at the end.
    std::string hostAddress = address;
    std::string last_element(hostAddress.substr(hostAddress.rfind(".")));
    auto ptr = hostAddress.rfind('.');
    hostAddress.replace(ptr, last_element.length(), ".2");

    std::string str = "Setting host address to: " + hostAddress;
    m_EventCallback(str, m_Context, 0);

    // CALL IOCTL Operations to set the address of the adapter/socket  //
    struct ifreq ifr{};
    /// note: no pointer here
    struct sockaddr_in inet_addr{}, subnet_mask{};
    // get interface name //
    // Prepare the struct ifreq //
    bzero(ifr.ifr_name, IFNAMSIZ);
    strncpy(ifr.ifr_name, adapter.networkAdapter.c_str(), IFNAMSIZ);

    /// note: prepare the two struct sockaddr_in
    inet_addr.sin_family = AF_INET;
    inet_pton(AF_INET, hostAddress.c_str(), &(inet_addr.sin_addr));

    subnet_mask.sin_family = AF_INET;
    inet_pton(AF_INET, "255.255.255.0", &(subnet_mask.sin_addr));


    // Call ioctl to configure network devices
    /// put addr in ifr structure
    memcpy(&(ifr.ifr_addr), &inet_addr, sizeof(struct sockaddr));
    int ioctl_result = ioctl(camera_fd, SIOCSIFADDR, &ifr);  // Set IP address
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFADDR: %s\n", strerror(errno));
        m_EventCallback("Error", m_Context, 0);
        return NO_CAMERA;
    }

    /// put mask in ifr structure
    memcpy(&(ifr.ifr_addr), &subnet_mask, sizeof(struct sockaddr));
    ioctl_result = ioctl(camera_fd, SIOCSIFNETMASK, &ifr);   // Set subnet mask
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFNETMASK: ");
        m_EventCallback("Error", m_Context, 0);
        return NO_CAMERA;
    }

    // Attempt to connect to camera and post some info
    str = "Checking for camera at: " + address;
    m_EventCallback(str, m_Context, 0);

    std::cout << "Camera interface: " << adapter.index << " name: " << adapter.networkAdapter << std::endl;
    if (interrupt) {
        shutdownT1Ready = true;
        return NO_CAMERA;
    }
}

void AutoConnectLinux::onFoundCamera() {
    m_Callback(result, m_Context);
}

void AutoConnectLinux::stopAutoConnect() {
    m_LoopAdapters = false;
    m_ListenOnAdapter = false;
    m_ShouldProgramRun = false;

    if (m_TAutoConnect != nullptr && shutdownT1Ready) {
        m_TAutoConnect->join();
        delete m_TAutoConnect;
        m_TAutoConnect = nullptr;
    }
    if (m_TAdapterSearch != nullptr && shutdownT2Ready) {
        m_TAdapterSearch->join();
        delete m_TAdapterSearch;
        m_TAdapterSearch = nullptr;
    }
}

void AutoConnectLinux::start() {
    m_LoopAdapters = true;
    m_ListenOnAdapter = true;
    m_ShouldProgramRun = true;
    if (m_TAutoConnect == nullptr)
        m_TAutoConnect = new std::thread(&AutoConnectLinux::run, this);
    else {
        interrupt = true;
        stopAutoConnect();
        m_LoopAdapters = true;
        m_ListenOnAdapter = true;
        m_ShouldProgramRun = true;
        while (true) {
            {
                std::scoped_lock<std::mutex> lock(readSupportedAdaptersMutex);
                if (shutdownT1Ready)
                    break;
            }
        }
        interrupt = false;
        m_TAutoConnect = new std::thread(&AutoConnectLinux::run, this);
    }

}


void AutoConnectLinux::startAdapterSearch() {
    if (m_TAdapterSearch == nullptr && m_ShouldProgramRun) {
        m_RunAdapterSearch = true;
        m_TAdapterSearch = new std::thread(&AutoConnectLinux::findEthernetAdapters, this, false, false,
                                           &supportedAdapters);
    }
}


void AutoConnectLinux::setDetectedCallback(void (*param)(Result result1, void *ctx), void *_context) {
    m_Callback = param;
    m_Context = _context;
}

bool AutoConnectLinux::isRunning() {
    return m_ShouldProgramRun; // Note: This is just confusing usage... future technical debt right here
}

void AutoConnectLinux::setShouldProgramRun(bool close) {
    this->m_ShouldProgramRun = close; // Note: This is just confusing usage... future technical debt right here
}

void AutoConnectLinux::setEventCallback(void (*param)(const std::string &str, void *, int)) {
    m_EventCallback = param;

}

void AutoConnectLinux::clearSearchedAdapters() {
    m_IgnoreAdapters.clear();
}
 */

