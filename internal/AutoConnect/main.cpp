//
// Created by magnus on 7/14/22.
//


#include <cstdlib>
#include "AutoConnect.h"

#ifdef WIN32
#include "AutoConnectWindows.h"
#else

#include "AutoConnectLinux.h"

#endif

int main() {


#ifdef WIN32
    AutoConnectWindows connect{};
    std::vector<AutoConnect::AdapterSupportResult> res =  connect.findEthernetAdapters(false);
#else
    AutoConnectLinux connect{};
    std::vector<AutoConnect::AdapterSupportResult> res = connect.findEthernetAdapters();
#endif

    connect.start(res);


    while (true) {

        if (connect.isConnected())
            break;
    }

    connect.stop();

    AutoConnect::Result result = connect.getResult();
    printf("Found Camera on IP: %s\n", result.cameraIpv4Address.c_str());
    printf("Using Adapter: %s -->", result.networkAdapter.c_str());
    printf("Long name: %s\n", result.networkAdapterLongName.c_str());
    crl::multisense::Channel *ptr = connect.getCameraChannel();
    crl::multisense::system::DeviceInfo info;

    ptr->getDeviceInfo(info);
    printf("Connected camera: %s with Imager: %s\n", info.name.c_str(), info.imagerName.c_str());


    return EXIT_SUCCESS;
}