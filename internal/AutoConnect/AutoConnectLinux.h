//
// Created by magnus on 7/14/22.
//

#ifndef AUTOCONNECT_AUTOCONNECTLINUX_H
#define AUTOCONNECT_AUTOCONNECTLINUX_H


#include <thread>
#include "AutoConnect.h"

class AutoConnectLinux : AutoConnect{

public:

    /** @Brief Starts the search for camera given a list containing network adapters Search is done in another thread**/
    void start(std::vector<AdapterSupportResult> vector) override;
    /** @Brief Function to search for network adapters **/
    std::vector<AdapterSupportResult> findEthernetAdapters() override;

    /** @Brief cleans up thread**/
    void stop() override;

    /** @Brief Function called after a search of adapters and at least one adapter was found **/
    void onFoundAdapters(std::vector<AdapterSupportResult> vector) override;
    /** @Brief Function called when a new IP is found. Return false if you want to keep searching or true to stop further IP searching **/
    FoundCameraOnIp onFoundIp(std::string string, AdapterSupportResult adapter) override;
    /** @Brief Function called when a camera has been found by a successfully connection by LibMultiSense **/
    void onFoundCamera(AdapterSupportResult supportResult) override;
    /** @Brief boolean set to true in onFoundCamera() **/
    bool isConnected() { return success; }

    AutoConnect::Result getResult();
    crl::multisense::Channel* getCameraChannel();
private:
    static void run(void* instance, std::vector<AdapterSupportResult> adapters);
};


#endif //AUTOCONNECT_AUTOCONNECTLINUX_H
