/**
 * @file: MultiSense-Viewer/include/Viewer/Tools/AdapterUtils.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-11-29, mgjerde@carnegierobotics.com, Created file.
 **/

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

#define LIST_ADAPTER_INTERVAL_MS 500

class AdapterUtils {
public:
    struct Adapter {
        std::string ifName;
        uint32_t ifIndex = 0;
        std::string description;
        bool supports = true;

        Adapter(std::string name, uint32_t index) : ifName(std::move(name)), ifIndex(index) {
        }

        Adapter() = default;

    };



    std::vector<Adapter> getAdaptersList() {
#ifdef WIN32
        std::scoped_lock<std::mutex> lock(mut);
        return adapters;
#else
        return listAdapters();
#endif
    }

    void startAdapterScan(VkRender::ThreadPool *pool) {
#ifdef WIN32
        std::scoped_lock<std::mutex> lock(mut);
        if (!runThread && !isThreadAlive) {
            pool->Push(AdapterUtils::listAdapters, &adapters, this);
            runThread = true;
            Log::Logger::getInstance()->info("Started Manual Adapter Scan");
        }
#endif
    }

    void stopAdapterScan() {
#ifdef WIN32
        std::scoped_lock<std::mutex> lock(mut);
        if (runThread) {
            runThread = false;
            Log::Logger::getInstance()->info("Sent stop to manual adapter scan thread");
        }
#endif
    }

    bool shutdownReady(){
        std::scoped_lock<std::mutex> lock(mut);
        return !isThreadAlive;
    }

#ifdef WIN32
    /**
     * Threaded function to list the connected adapter
     * All subsequent calls checks if there is any found adapters
     * starts execution by calling startAdapterScan
     * Stops execution by calling stopAdapterScan()
     * @return
     */
    static void listAdapters(std::vector<Adapter> *adapters, AdapterUtils *ctx) {
        {
            std::scoped_lock<std::mutex> lock(ctx->mut);
            ctx->isThreadAlive = true;
        }
        while (true) {
            {
                std::scoped_lock<std::mutex> lock(ctx->mut);
                if (!ctx->runThread)
                    break;
            }
            Log::Logger::getInstance()->trace("Listing connected adapters for Manual connect");

            pcap_if_t *alldevs;
            pcap_if_t *d;
            char errbuf[PCAP_ERRBUF_SIZE];

            // Retrieve the device list
            if (pcap_findalldevs(&alldevs, errbuf) == -1) {
                Log::Logger::getInstance()->error("Failed to list adapters in pcap_findalldevs in manual connect. Stopping");
                std::this_thread::sleep_for(std::chrono::milliseconds(LIST_ADAPTER_INTERVAL_MS));
                break;
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
            pcap_freealldevs(alldevs);

            DWORD dwBufLen = sizeof(IP_ADAPTER_INFO);
            PIP_ADAPTER_INFO AdapterInfo;
            AdapterInfo = (IP_ADAPTER_INFO *) malloc(sizeof(IP_ADAPTER_INFO));
            // Make an initial call to GetAdaptersInfo to get the necessary size into the dwBufLen variable
            if (GetAdaptersInfo(AdapterInfo, &dwBufLen) == ERROR_BUFFER_OVERFLOW) {
                free(AdapterInfo);
                AdapterInfo = (IP_ADAPTER_INFO *) malloc(dwBufLen);
                if (AdapterInfo == NULL) {
                    free(AdapterInfo);
                    Log::Logger::getInstance()->error("Failed to GetAdaptersInfo");
                    std::this_thread::sleep_for(std::chrono::milliseconds(LIST_ADAPTER_INTERVAL_MS));
                    continue;
                }
            }
            if (GetAdaptersInfo(AdapterInfo, &dwBufLen) == NO_ERROR) {
                // Contains pointer to current adapter info
                std::scoped_lock<std::mutex> lock(ctx->mut);
                adapters->clear();

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
                    size_t lenA = strlen(prefix.c_str());
                    size_t lenB = strlen(pAdapterInfo->AdapterName);
                    char *con = (char *) malloc(lenA + lenB + 1);
                    memcpy(con, prefix.c_str(), lenA);
                    memcpy(con + lenA, pAdapterInfo->AdapterName, lenB + 1);
                    adapter.ifName = con;
                    //adapter.networkAdapter = pAdapterInfo->AdapterName;
                    adapter.ifIndex = pAdapterInfo->Index;
                    adapters->push_back(adapter);
                    free(con);
                    pAdapterInfo = pAdapterInfo->Next;
                } while (pAdapterInfo);
            }
            free(AdapterInfo);
            std::this_thread::sleep_for(std::chrono::milliseconds(LIST_ADAPTER_INTERVAL_MS));
        }

        std::scoped_lock<std::mutex> lock(ctx->mut);
        ctx->isThreadAlive = false;
        Log::Logger::getInstance()->info("Manual Adapter Scan task finished");
    }

#else

    static inline std::vector<Adapter> listAdapters() {
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

private:
    std::vector<Adapter> adapters;
    std::mutex mut;
    bool runThread = false;
    bool isThreadAlive = false;
};

#endif //MULTISENSE_VIEWER_ADAPTERUTILS_H
