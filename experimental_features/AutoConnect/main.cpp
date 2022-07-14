//
// Created by magnus on 7/14/22.
//


#include <cstdlib>
#include "AutoConnectLinux.h"
#include "AutoConnect.h"

int main() {

    AutoConnectLinux connect{};

    std::vector<AutoConnect::AdapterSupportResult> res =  connect.findEthernetAdapters();


    connect.start(res);


    while (true){

        if (connect.success)
            break;
    }

    connect.stop();


    return EXIT_SUCCESS;
}