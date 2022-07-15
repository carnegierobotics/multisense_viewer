//
// Created by magnus on 7/14/22.
//


#include <cstdlib>
#include "AutoConnectWindows.h"
#include "AutoConnect.h"

int main() {

    AutoConnectWindows connect{};

    std::vector<AutoConnect::AdapterSupportResult> res =  connect.findEthernetAdapters();


    connect.start(res);


    while (true){

        if (connect.isConnected())
            break;
    }

    connect.stop();

    AutoConnect::Result result = connect.getResult();
    printf("Found Camera on IP: %s\n", result.cameraIpv4Address.c_str());
    printf("Using Adapter: %s -->", result.networkAdapter.c_str());
    printf("Long name: %s\n", result.networkAdapterLongName.c_str());
    crl::multisense::Channel * ptr = connect.getCameraChannel();
    crl::multisense::system::DeviceInfo info;

    ptr->getDeviceInfo(info);
    printf("Connected camera: %s with Imager: %s\n", info.name.c_str(), info.imagerName.c_str());




    return EXIT_SUCCESS;
}