//
// Created by mgjer on 14/07/2022.
//
#ifdef _MSC_VER
/*
 * we do not want the warnings about the old deprecated and unsecure CRT functions
 * since these examples can be compiled under *nix as well
 */
#define _CRT_SECURE_NO_WARNINGS
#endif
#pragma comment(lib, "wpcap.lib")
#pragma comment(linker, "/SUBSYSTEM:CONSOLE")
#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "ws2_32.lib")



#define MALLOC(x) HeapAlloc(GetProcessHeap(), 0, (x))
#define FREE(x) HeapFree(GetProcessHeap(), 0, (x))
#define ADAPTER_HEX_NAME_LENGTH 38
#define UNNAMED_ADAPTER "Unnamed"

#include <winsock2.h>
#include <iphlpapi.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <iprtrmib.h>
#include <time.h>

#include "AutoConnectWindows.h"
#include "pcap.h"

struct iphdr {
    unsigned char ip_verlen;            // 4-bit IPv4 version, 4-bit header length (in 32-bit words)
    unsigned char ip_tos;                 // IP type of service
    unsigned short ip_totallength;    // Total length
    unsigned short ip_id;                  // Unique identifier
    unsigned short ip_offset;            // Fragment offset field
    unsigned char ip_ttl;                   // Time to liv
    unsigned char protocol;       // Protocol(TCP,UDP etc)
    unsigned short ip_checksum;    // IP checksum
    unsigned int saddr;           // Source address
    unsigned int ip_destaddr;         // Source address

};

std::vector<AdapterSupportResult> AutoConnectWindows::findEthernetAdapters(bool b) {
    std::vector<AutoConnect::AdapterSupportResult> adapters;
    pcap_if_t *alldevs;
    pcap_if_t *d;
    char errbuf[PCAP_ERRBUF_SIZE];

    /* Retrieve the device list */
    if (pcap_findalldevs(&alldevs, errbuf) == -1) {
        fprintf(stderr, "Error in pcap_findalldevs: %s\n", errbuf);
    }

    /* Print the list */
    int i = 0;
    for (d = alldevs; d; d = d->next) {
        AdapterSupportResult adapter(UNNAMED_ADAPTER, 0);
        adapter.lName = d->name;
        if (d->description)
            adapter.description = d->description;

        adapters.emplace_back(adapter);

    }

    /** REST OF FUNCTION TRIES TO RENAME THE ADAPTERS FOUND INTO USABLE NAMES **/
    struct PowerShellOutput {
        std::string name;
        std::string tplName;
    };
    std::vector<PowerShellOutput> names;

    // Create Readable names from windows' hex string idetifiers
    std::ofstream psFile{};
    std::string psName = "tmp.ps1";
    std::string tmpTxtFile = "tmp.txt";
    psFile.open(psName);
    std::string powershell;
    powershell = "getmac /v /fo csv > " + tmpTxtFile + " \n";
    powershell +=
            "((Get-Content " + tmpTxtFile + ")  -replace '`n','\n' -replace '`r','\n')  | Set-Content " + tmpTxtFile +
            "\n";
    psFile << powershell << std::endl;
    psFile.close();

    system("start powershell.exe Set-ExecutionPolicy RemoteSigned \n");
    system((std::string("powershell.exe ") + "-File " + psName).c_str());
    system("cls");
    remove(psName.c_str());

    std::ifstream resfile(tmpTxtFile);
    std::string line;
    if (resfile.is_open()) {
        while (getline(resfile, line)) {
            char *token = strtok(line.data(), ",");
            int i = 0;
            PowerShellOutput psOutput;
            while (token != nullptr) {
                if (i == 0)
                    psOutput.name = token;
                if (i == 3) {
                    std::string tpl(token);
                    tpl.replace(tpl.length() - 1, tpl.length(), "");
                    psOutput.tplName = tpl;
                }
                i++;
                token = strtok(nullptr, ",");
            }
            names.emplace_back(psOutput);

        }
        resfile.close();
        remove(tmpTxtFile.c_str());
    }

    // Match hex strings.
    // Loop through the found adapters and cross check if any of the hex strings are simillar to the getmacs commands from powershell


    for (auto &psOut: names) {
        if (psOut.tplName.length() < ADAPTER_HEX_NAME_LENGTH)
            continue;
        std::string tplName = psOut.tplName.substr(psOut.tplName.size() - ADAPTER_HEX_NAME_LENGTH);

        for (auto &adapter: adapters) {
            if (adapter.lName.length() < ADAPTER_HEX_NAME_LENGTH && adapter.name != UNNAMED_ADAPTER)
                continue;
            std::string hexString = adapter.lName.substr(adapter.lName.size() - ADAPTER_HEX_NAME_LENGTH);

            if (hexString == tplName) {
                adapter.name = psOut.name;
                adapter.supports = true;

            }
        }

    }

    if (!adapters.empty())
        onFoundAdapters(adapters);

    return adapters;
}

void AutoConnectWindows::start(std::vector<AdapterSupportResult> adapters) {
    //run(this, adapters);
    t = new std::thread(&AutoConnectWindows::run, this, adapters);
    printf("Started thread\n");
}

void AutoConnectWindows::onFoundAdapters(std::vector<AdapterSupportResult> vector) {

}


AutoConnect::FoundCameraOnIp
AutoConnectWindows::onFoundIp(std::string cameraAddress, AutoConnect::AdapterSupportResult adapter) {
    DWORD dwSize = 0, dwRetVal = 0;
    // Before calling AddIPAddress we use GetIpAddrTable to get
    // an adapter to which we can add the IP.
    PMIB_IPADDRTABLE pIPAddrTable = (MIB_IPADDRTABLE *) MALLOC(sizeof(MIB_IPADDRTABLE));
    if (pIPAddrTable == NULL) {
        printf("Error allocating memory needed to call GetIpAddrTable\n");
        exit(1);
    } else {
        dwSize = 0;
        // Make an initial call to GetIpAddrTable to get the
        // necessary size into the dwSize variable
        if (GetIpAddrTable(pIPAddrTable, &dwSize, 0) ==
            ERROR_INSUFFICIENT_BUFFER) {
            FREE(pIPAddrTable);
            pIPAddrTable = (MIB_IPADDRTABLE *) MALLOC(dwSize);

        }
        if (pIPAddrTable == NULL) {
            printf("Memory allocation failed for GetIpAddrTable\n");
            exit(1);
        }
    }
    std::string hostAddress = cameraAddress;
    hostAddress.replace(cameraAddress.rfind("."), cameraAddress.length(), ".2");

    DWORD ifIndex;
    IN_ADDR IPAddr;
    // Make a second call to GetIpAddrTable to get the
    // actual data we want
    if ((dwRetVal = GetIpAddrTable(pIPAddrTable, &dwSize, 0)) == NO_ERROR) {
        // Save the interface index to use for adding an IP address
        ifIndex = pIPAddrTable->table[0].dwIndex;
        printf("\n\tInterface Index:\t%ld\n", ifIndex);
        IPAddr.S_un.S_addr = (u_long) pIPAddrTable->table[0].dwAddr;

        if (inet_ntoa(IPAddr) == hostAddress) { hostAddress = "10.66.171.20"; }

        printf("\tIP Address:       \t%s (%lu%)\n", inet_ntoa(IPAddr),
               pIPAddrTable->table[0].dwAddr);
        IPAddr.S_un.S_addr = (u_long) pIPAddrTable->table[0].dwMask;
        printf("\tSubnet Mask:      \t%s (%lu%)\n", inet_ntoa(IPAddr),
               pIPAddrTable->table[0].dwMask);
        IPAddr.S_un.S_addr = (u_long) pIPAddrTable->table[0].dwBCastAddr;
        printf("\tBroadCast Address:\t%s (%lu%)\n", inet_ntoa(IPAddr),
               pIPAddrTable->table[0].dwBCastAddr);
        printf("\tReassembly size:  \t%lu\n\n",
               pIPAddrTable->table[0].dwReasmSize);

    } else {
        printf("Call to GetIpAddrTable failed with error %d.\n", dwRetVal);
        if (pIPAddrTable)
            FREE(pIPAddrTable);
        exit(1);
    }

    if (pIPAddrTable) {
        FREE(pIPAddrTable);
        pIPAddrTable = NULL;
    }

    /* Variables where handles to the added IP are returned */
    ULONG NTEContext = 0;
    ULONG NTEInstance = 0;
    // Attempt to connect to camera and post some info

    unsigned long ulAddr = inet_addr(hostAddress.c_str());
    unsigned long ulMask = inet_addr("255.255.255.0");
    LPVOID lpMsgBuf;

    if ((dwRetVal = AddIPAddress(ulAddr,
                                 ulMask,
                                 ifIndex,
                                 &NTEContext, &NTEInstance)) == NO_ERROR) {
        printf("\tIPv4 address %s was successfully added.\n", hostAddress.c_str());
    } else {
        printf("AddIPAddress failed with error: %d\n", dwRetVal);
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                          NULL, dwRetVal, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),       // Default language
                          (LPTSTR) &lpMsgBuf, 0, NULL)) {
            printf("\tError: %s", lpMsgBuf);
            LocalFree(lpMsgBuf);
        }
    }
    cameraInterface = crl::multisense::Channel::Create(cameraAddress);

    if (cameraInterface == nullptr && connectAttemptCounter > MAX_CONNECTION_ATTEMPTS) {
        connectAttemptCounter = 0;
        return NO_CAMERA;
    } else if (cameraInterface == nullptr) {
        connectAttemptCounter++;
        return NO_CAMERA_RETRY;
    } else {
        result.networkAdapter = adapter.name;
        result.networkAdapterLongName = adapter.lName;
        result.cameraIpv4Address = cameraAddress;
        return FOUND_CAMERA;
    }


}

void AutoConnectWindows::onFoundCamera(AdapterSupportResult supportResult) {

}

void AutoConnectWindows::stop() {
    printf("Joining Thread\n");
    t->join();


}

void AutoConnectWindows::run(void *instance, std::vector<AdapterSupportResult> adapters) {
    auto *app = (AutoConnectWindows *) instance;

    // Get list of network adapters that are  supports our application
    std::string hostAddress;
    size_t i = adapters.size();

    // Loop keeps retrying to connect on supported network adapters.
    while (app->loopAdapters) {
        if (i == 0)
            i = adapters.size();
        i--;
        auto adapter = adapters[i];

        if (!adapter.supports) {
            continue;
        }
        printf("\nTesting Adapter: %s\n", adapter.name.c_str());
        app->startTime = time(nullptr);

        pcap_if_t *alldevs;
        pcap_t *adhandle;
        int res;
        char errbuf[PCAP_ERRBUF_SIZE];
        struct tm *ltime;
        char timestr[16];
        struct pcap_pkthdr *header;
        const u_char *pkt_data;
        time_t local_tv_sec;

        /* Open the adapter */
        if ((adhandle = pcap_open_live(adapter.lName.c_str(),    // name of the device
                                       65536,            // portion of the packet to capture.
                // 65536 grants that the whole packet will be captured on all the MACs.
                                       1,                // promiscuous mode (nonzero means promiscuous)
                                       1000,            // read timeout
                                       errbuf            // error buffer
        )) == NULL) {
            fprintf(stderr, "\nUnable to open the adapter. %s is not supported by WinPcap\n", adapter.name.c_str());
            /* Free the device list */
            return;
        }

        printf("\nlistening on %s...\n", adapter.name.c_str());

        std::string ip;

        while (app->listenOnAdapter) {

            // Timeout handler
            if ((time(nullptr) - app->startTime) > TIMEOUT_INTERVAL_SECONDS){
                app->startTime = time(nullptr);
                printf("Timeout reached. Switching adapter\n");
                break;
            }

            res = pcap_next_ex(adhandle, &header, &pkt_data);

            if (res == 0)
                /* Timeout elapsed */
                continue;
            /* convert the timestamp to readable format */
            local_tv_sec = header->ts.tv_sec;
            ltime = localtime(&local_tv_sec);
            strftime(timestr, sizeof timestr, "%H:%M:%S", ltime);

            /* retireve the position of the ip header */
            auto *ih = (iphdr *) (pkt_data +
                                   14); //length of ethernet header

            char ips[16];
            sprintf(ips, "%d.%d.%d.%d", (ih->saddr >> (8 * 0)) & 0xff,
                    (ih->saddr >> (8 * 1)) & 0xff,
                    (ih->saddr >> (8 * 2)) & 0xff,
                    (ih->saddr >> (8 * 3)) & 0xff);

            ip = ips;

            if (ih->protocol == 2) {
                printf("%s: Source ip: %d.%d.%d.%d, Protocol: %d, Len:%d --> ", timestr, (ih->saddr >> (8 * 0)) & 0xff,
                       (ih->saddr >> (8 * 1)) & 0xff,
                       (ih->saddr >> (8 * 2)) & 0xff,
                       (ih->saddr >> (8 * 3)) & 0xff, ih->protocol, header->len);

                printf("Found IGMP packet. Source IP could be camera\n ");
                FoundCameraOnIp ret = app->onFoundIp(ips, adapter);

                if (ret == FOUND_CAMERA) {
                    std::cout << "Found camera. quitting..." << std::endl;
                    app->listenOnAdapter = false;
                    app->loopAdapters = false;
                    app->success = true;
                    break;
                } else if (ret == NO_CAMERA_RETRY) {
                    std::cout << "No camera retrying on same adapter" << std::endl;
                    continue;
                } else if (ret == NO_CAMERA) {
                    std::cout << "No camera trying other adapter" << std::endl;
                    break;
                }
            }
        }
    }
}

AutoConnect::Result AutoConnectWindows::getResult() {
    return result;
}

crl::multisense::Channel* AutoConnectWindows::getCameraChannel() {
    return cameraInterface;
}

