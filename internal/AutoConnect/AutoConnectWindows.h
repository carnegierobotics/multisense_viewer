//
// Created by mgjer on 14/07/2022.
//

#ifndef AUTOCONNECT_AUTOCONNECTWINDOWS_H
#define AUTOCONNECT_AUTOCONNECTWINDOWS_H


#include "AutoConnect.h"

class AutoConnectWindows : AutoConnect {


public:

    /** @Brief Starts the search for camera given a list containing network adapters Search is done in another thread**/
    void start(std::vector<Result> vector) override;
    /** @Brief Function to search for network adapters **/
    std::vector<AutoConnect::Result> findEthernetAdapters(bool b, bool skipIgnored) override;

    /** @Brief cleans up thread**/
    void stop() override;

    /** @Brief Function called after a search of adapters and at least one adapter was found **/
    void onFoundAdapters(std::vector<Result> vector, bool logEvent) override;
    /** @Brief Function called when a new IP is found. Return false if you want to keep searching or true to stop further IP searching **/
    AutoConnect::FoundCameraOnIp onFoundIp(std::string address, Result adapter, int camera_fd) override;
    /** @Brief Function called when a camera has been found by a successfully connection by LibMultiSense **/
    void onFoundCamera(Result supportResult) override;
    /** @Brief boolean set to true in onFoundCamera() **/
    bool isConnected() { return success; }

    AutoConnect::Result getResult();
    crl::multisense::Channel* getCameraChannel();

    void setDetectedCallback(void (*param)(Result result1, void* ctx), void* context);
    void setEventCallback(void (*param)(std::string result1, void* ctx, int));

    void (*callback)(AutoConnect::Result, void*) = nullptr;
    void (*eventCallback)(std::string, void*, int) = nullptr;


    void* context = nullptr;
    bool running = false;
    bool shouldProgramClose() override;
    void setShouldProgramClose(bool close) override;

    void clearSearchedAdapters();
    void disableAutoConfiguration(std::string ip, uint32_t index);

private:
    static void run(void* instance, std::vector<Result> adapters);


};


#endif //AUTOCONNECT_AUTOCONNECTWINDOWS_H
