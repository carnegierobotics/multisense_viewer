//
// Created by magnus on 7/14/22.
//

#ifndef AUTOCONNECT_AUTOCONNECTLINUX_H
#define AUTOCONNECT_AUTOCONNECTLINUX_H


#include <thread>
#include "AutoConnect.h"

class AutoConnectLinux : AutoConnect{


public:
    void start(std::vector<AdapterSupportResult> vector) override;

    std::vector<AdapterSupportResult> findEthernetAdapters() override;

    void onFoundAdapters(std::vector<AdapterSupportResult> vector) override;

    /** @Brief Function called when a new IP is found. Return false if you want to keep searching or true to stop further IP searching **/
    bool onFoundIp(std::string address, AdapterSupportResult adapter) override;

    void onFoundCamera() override;

    void stop() override;

    bool success = false;
    bool loopAdapters = true;
    bool listenOnAdapter = true;

protected:
    static void run(void *instance, std::vector<AdapterSupportResult> adapters);


private:
    std::thread *t;
};


#endif //AUTOCONNECT_AUTOCONNECTLINUX_H
