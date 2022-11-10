//
// Created by magnus on 7/14/22.
//

#ifndef AUTOCONNECT_AUTOCONNECTLINUX_H
#define AUTOCONNECT_AUTOCONNECTLINUX_H


#include <thread>
#include "AutoConnect.h"
#include <mutex>

class AutoConnectLinux : AutoConnect{

public:

    ~AutoConnectLinux() {
        m_LoopAdapters = false;
        m_ListenOnAdapter = false;
        m_ShouldProgramRun = false;
        m_RunAdapterSearch = false;

        if (m_TAutoConnect != nullptr) {
            m_TAutoConnect->join();
            delete m_TAutoConnect;
            shutdownT1Ready = false;
            m_TAutoConnect = nullptr;
        }
        if (m_TAdapterSearch != nullptr) {
            m_TAdapterSearch->join();
            delete m_TAdapterSearch;
            shutdownT2Ready = false;
            m_TAdapterSearch = nullptr;
        }
    }

    /**
     * @Brief Starts the search for camera given a list containing network adapters Search is done in another thread
    * @param vector
     */
    void start() override;
    /** @Brief Function to search for network adapters **/
    static void findEthernetAdapters(void *ctx, bool logEvent, bool skipIgnored,
                                     std::vector<AutoConnect::Result> *res);
    /** @Brief cleans up thread**/
    void stopAutoConnect() override;
    /** @Brief Function called after a search of adapters and at least one adapter was found **/
    void onFoundAdapters(std::vector<Result> vector, bool logEvent) override;
    /** @Brief Function called when a new IP is found. Return false if you want to keep searching or true to stop further IP searching **/
    AutoConnect::FoundCameraOnIp onFoundIp(std::string address, Result adapter, int camera_fd) override;
    /** @Brief Function called when a camera has been found by a successfully connection by LibMultiSense **/
    void onFoundCamera() override;

    crl::multisense::Channel* getCameraChannel();

    void setDetectedCallback(void (*param)(Result result1, void* ctx), void* context);
    void setEventCallback(void (*param)(const std::string& result1, void* ctx, int));

    void (*m_Callback)(AutoConnect::Result, void*) = nullptr;
    void (*m_EventCallback)(const std::string&, void*, int) = nullptr;
    bool m_RunAdapterSearch = true;

    std::vector<AutoConnect::Result> supportedAdapters;

    void* m_Context = nullptr;
    bool isRunning() override;
    void setShouldProgramRun(bool close) override;
    std::mutex readSupportedAdaptersMutex;

    bool shutdownT1Ready = false;
    bool shutdownT2Ready = false;

    void clearSearchedAdapters();

    void startAdapterSearch();

private:
    static void run(void* instance);
};


#endif //AUTOCONNECT_AUTOCONNECTLINUX_H
