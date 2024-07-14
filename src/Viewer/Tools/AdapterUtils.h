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
#ifdef WIN32
#else
#include <net/if.h>
#include <netinet/in.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif
#include "Viewer/Tools/Macros.h"
#include "Viewer/Tools/ThreadPool.h"
#include "Viewer/Tools/Logger.h"

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

    AdapterUtils() {
        m_runAdapterScan = true;
        m_pool = std::make_unique<VkRender::ThreadPool>(1);
        m_pool->Push(AdapterUtils::startAdapterScan, this);
        Log::Logger::getInstance()->info("Started Manual Adapter Scan");
    }

    ~AdapterUtils(){
        Log::Logger::getInstance()->trace("Stopping the manual Adapter Scan");
        m_runAdapterScan = false;
        m_pool->Stop();
        Log::Logger::getInstance()->trace("Stopped the manual Adapter Scan");
    }

    void setAdapterList(std::vector<Adapter> adapters) {
        std::scoped_lock<std::mutex> lock(m_adapterMutex);
        m_adapterList = std::move(adapters);
    }

    std::vector<Adapter> getAdapterList() {
        std::scoped_lock<std::mutex> lock(m_adapterMutex);
        return m_adapterList;
    }

private:
    std::vector<Adapter> m_adapterList;
    std::mutex m_adapterMutex;
    bool m_runAdapterScan = false;
    std::unique_ptr<VkRender::ThreadPool> m_pool;

    static void startAdapterScan(void *ctx) {
        auto *context = reinterpret_cast<AdapterUtils *>(ctx);
        while (context->m_runAdapterScan) {
            auto adapters = listAdapters();
            context->setAdapterList(adapters);
        }
    }


    static inline std::vector<Adapter> listAdapters() {
        // Get list of interfaces
        std::vector<Adapter> adapters;

        #ifdef WIN32
#else
        auto ifn = if_nameindex();
        // If no interfaces. This turns to null if there is no interfaces for a few seconds
        if (!ifn) {
            Log::Logger::getInstance()->error("Failed to list adapters: if_nameindex(), strerror: %s", strerror(errno));
        }
        auto fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
        for (auto i = ifn; i->if_name; ++i) {
            struct {
                DISABLE_WARNING_PUSH
                DISABLE_WARNING_PEDANTIC
                struct ethtool_link_settings req{};
                DISABLE_WARNING_POP
                __u32 link_mode_data[3 * 127]{};
            } ecmd{};
            Adapter adapter(i->if_name, i->if_index);

            auto ifr = ifreq{};
            std::strncpy(ifr.ifr_name, i->if_name, IF_NAMESIZE);
            ifr.ifr_name[IF_NAMESIZE - 1] = '\0';
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
#endif
        return adapters;
    }


};

#endif //MULTISENSE_VIEWER_ADAPTERUTILS_H
