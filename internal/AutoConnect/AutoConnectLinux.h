//
// Created by magnus on 7/14/22.
//

#ifndef AUTOCONNECT_AUTOCONNECTLINUX_H
#define AUTOCONNECT_AUTOCONNECTLINUX_H


#include <thread>
#include "AutoConnect.h"

class AutoConnectLinux : AutoConnect{

public:

    /**
     * @Brief Starts the search for camera given a list containing network adapters Search is done in another thread
    * @param vector
     */
    void start(std::vector<Result> vector) override;
    /** @Brief Function to search for network adapters **/
    std::vector<AutoConnect::Result> findEthernetAdapters(bool logEvent, bool skipIgnored) override;
    /** @Brief cleans up thread**/
    void stop() override;
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


    void* m_Context = nullptr;
    bool isRunning() override;
    void setShouldProgramRun(bool close) override;


    void clearSearchedAdapters();

private:
    static void run(void* instance, std::vector<Result> adapters);
};


#endif //AUTOCONNECT_AUTOCONNECTLINUX_H
