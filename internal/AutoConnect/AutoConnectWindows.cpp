#include "AutoConnectWindows.h"
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

#define _WINSOCKAPI_    // stops windows.h including winsock.h
#include <WinSock2.h>
#include <iphlpapi.h>

#define _WIN32_DCOM
#include <comdef.h>
#include <Wbemidl.h>

#pragma comment(lib, "wbemuuid.lib")

#include <cstdio>
#include <fstream>
#include <iostream>
#include <Iprtrmib.h>
#include <ctime>
#include <filesystem>

#include "AutoConnectWindows.h"
#include "pcap.h"

typedef struct IP_INFO
{
    std::string ip;
    std::string netmask;
} IP_INFO, * PIP_INFO;

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

std::vector<AutoConnect::Result>
AutoConnectWindows::findEthernetAdapters(bool logEvent, bool skipIgnored) {
    std::vector<AutoConnect::Result> adapters;
   
    
    pcap_if_t* alldevs;
    pcap_if_t* d;
    char errbuf[PCAP_ERRBUF_SIZE];

    /* Retrieve the device list */
    if (pcap_findalldevs(&alldevs, errbuf) == -1) {
        std::cout << "Error in pcap_findalldevs: " << errbuf << std::endl;
    }


    // Print the list
    int i = 0;
    std::string prefix = "\\Device\\Tcpip_";
    const char* token = "{";
    // Find { token in order to find correct prefix
    for (d = alldevs; d; d = d->next) {
        Result adapter(UNNAMED_ADAPTER, 0);
        adapter.networkAdapterLongName = d->name;
        if (d->description)
            adapter.description = d->description;

        std::cout << d->name << std::endl;

        size_t found = std::string(d->name).find(token);

        if (found != std::string::npos)
            prefix = std::string(d->name).substr(0, found);


    }

    std::cout << "prefix: " << prefix << std::endl;
    
    //
    /** REST OF FUNCTION TRIES TO RENAME THE ADAPTERS FOUND INTO USABLE NAMES 
    struct PowerShellOutput {
        std::string name;
        std::string tplName;
    };
    std::vector<PowerShellOutput> names;

    // Create Readable names from windo string idetifiers
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
            char* token = strtok(line.data(), ",");
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

    // Ma strings.
    // Loop through the found adapters and cross check if any of  strings are simillar to the getmacs commands from powershell


    for (auto& psOut : names) {
        if (psOut.tplName.length() < ADAPTER_HEX_NAME_LENGTH)
            continue;
        std::string tplName = psOut.tplName.substr(psOut.tplName.size() - ADAPTER_HEX_NAME_LENGTH);

        for (auto& adapter : adapters) {
            if (adapter.lName.length() < ADAPTER_HEX_NAME_LENGTH && adapter.name != UNNAMED_ADAPTER)
                continue;
            std::strString = adapter.lName.substr(adapter.lName.size() - ADAPTER_HEX_NAME_LENGTH);

            String == tplName) {
                adapter.name = psOut.name;
                adapter.supports = true;

            }
        }

    }
    */
    /* Another attempt at making the wind string user readable*/

    
    PIP_ADAPTER_INFO AdapterInfo;
    DWORD dwBufLen = sizeof(IP_ADAPTER_INFO);
    char* mac_addr = (char*)malloc(18);

    AdapterInfo = (IP_ADAPTER_INFO*)malloc(sizeof(IP_ADAPTER_INFO));
    if (AdapterInfo == NULL) {
        printf("Error allocating memory needed to call GetAdaptersinfo\n");
        free(mac_addr);
    }

    // Make an initial call to GetAdaptersInfo to get the necessary size into the dwBufLen variable
    if (GetAdaptersInfo(AdapterInfo, &dwBufLen) == ERROR_BUFFER_OVERFLOW) {
        free(AdapterInfo);
        AdapterInfo = (IP_ADAPTER_INFO*)malloc(dwBufLen);
        if (AdapterInfo == NULL) {
            printf("Error allocating memory needed to call GetAdaptersinfo\n");
            free(mac_addr);
        }
    }

    if (GetAdaptersInfo(AdapterInfo, &dwBufLen) == NO_ERROR) {
        // Contains pointer to current adapter info
        PIP_ADAPTER_INFO pAdapterInfo = AdapterInfo;
        do {

            // Somehow The integrated bluetooth adapter is considered an ethernet adapter cause of the same type in PIP_ADAPTER_INFO field "MIB_IF_TYPE_ETHERNET"
            // I'll filter it out here assuming it has Bluetooth in its name. Just a soft error which increases running time of the auto connect feature

            char* bleFoundInName = strstr(pAdapterInfo->Description, "Bluetooth");
            
            if (bleFoundInName || pAdapterInfo->Type != MIB_IF_TYPE_ETHERNET){
                pAdapterInfo = pAdapterInfo->Next;
                continue;

            }

            // Internal loopback device always located at index = 1. Skip it..
            if (pAdapterInfo->Index == 1) {
                pAdapterInfo = pAdapterInfo->Next;
                continue;
            }

            Result adapter(UNNAMED_ADAPTER, 0);
            adapter.supports = (pAdapterInfo->Type == MIB_IF_TYPE_ETHERNET);
            adapter.description = pAdapterInfo->Description;
            
            /*CONCATENATE two strings safely*/
            int lenA = strlen(prefix.c_str());
            int lenB = strlen(pAdapterInfo->AdapterName);
            char* con = (char *) malloc(lenA + lenB + 1);
            memcpy(con, prefix.c_str(), lenA);
            memcpy(con + lenA, pAdapterInfo->AdapterName, lenB + 1);
            adapter.networkAdapter = con;
            adapter.index = pAdapterInfo->Index;


            // If a camera has al ready been found on the adapter then set the searched flag.
            bool ignore = false;
            for (const auto& found : ignoreAdapters) {
                if (found.index == adapter.index) {
                    ignore = true;
                    std::string str = "Found already searched adapter: " + adapter.networkAdapter + ". Ignoring...";
                    eventCallback(str, context, 0);
                }
            }

            if (ignore && !skipIgnored) {
                adapter.searched = true;
            }

            adapters.push_back(adapter);
            free(con);
            // technically should look at pAdapterInfo->AddressLength
            //   and not assume it is 6.

            printf("Address: %s, : %s : %s : Type: %d\n", pAdapterInfo->IpAddressList.IpAddress.String, pAdapterInfo->AdapterName, pAdapterInfo->Description, pAdapterInfo->Type);
            
            pAdapterInfo = pAdapterInfo->Next;
        } while (pAdapterInfo);
    }
    free(AdapterInfo);
    


    if (!adapters.empty())
        onFoundAdapters(adapters, logEvent);

    return adapters;
}

void AutoConnectWindows::start(std::vector<Result> adapters) {
    // TODO Clean up public booleans. 4 member booleans might be exaggerated use?
    running = true;
    loopAdapters = true;
    listenOnAdapter = true;
    shouldProgramRun = true;

    t = new std::thread(&AutoConnectWindows::run, this, adapters);
    printf("Started thread\n");
}

void AutoConnectWindows::onFoundAdapters(std::vector<Result> adapters, bool logEvent) {


    for (auto& adapter : adapters) {
        if (adapter.supports && !adapter.searched) {
            std::string str;
            str = "Found supported adapter: " + adapter.networkAdapter;
            if (logEvent)
                eventCallback(str, context, 1);
        }
    }

}


AutoConnect::FoundCameraOnIp AutoConnectWindows::onFoundIp(std::string address, Result adapter, int camera_fd) {
  
  
    std::string hostAddress(address);

    size_t it = hostAddress.rfind('.', hostAddress.length());
    hostAddress.replace(it, hostAddress.length(), ".2");

    std::string str = "Setting host address to: " + hostAddress;
    eventCallback(str, context, 0);



    /* Variables where handles to the added IP are returned */
    ULONG NTEContext = 0;
    ULONG NTEInstance = 0;
    // Attempt to connect to camera and post some info

    unsigned long ulAddr = inet_addr(hostAddress.c_str());
    unsigned long ulMask = inet_addr("255.255.255.0");
    LPVOID lpMsgBuf;
    DWORD dwRetVal;
    /*
    disable*/

    if ((dwRetVal = AddIPAddress(ulAddr,
        ulMask,
        adapter.index,
        &NTEContext, &NTEInstance)) == NO_ERROR) {
        printf("\tIPv4 address %s was successfully added.\n", hostAddress.c_str());
    }
    else {
        // 5010 might be ERROR_DUP_DOMAINNAME according to description of error, but 5010 means address already exists at that adapter
        if (dwRetVal == 5010) {
            printf("Ip: already set to: %s\n", address);
        } else {
            printf("AddIPAddress failed with error: %d\n", dwRetVal);
            if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, dwRetVal, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),       // Default language
                (LPTSTR)&lpMsgBuf, 0, NULL)) {
                printf("\tError: %s", lpMsgBuf);
                LocalFree(lpMsgBuf);
            }
        }
        }
        
    

    // Attempt to connect to camera and post some info
    str = "Checking for camera at: " + address;
    eventCallback(str, context, 0);

    cameraInterface = crl::multisense::Channel::Create(address);

    if (cameraInterface == nullptr && connectAttemptCounter >= MAX_CONNECTION_ATTEMPTS) {
        connectAttemptCounter = 0;
        return NO_CAMERA;
    }
    else if (cameraInterface == nullptr) {
        connectAttemptCounter++;
        return NO_CAMERA_RETRY;
    }
    else {
        result.networkAdapter = adapter.networkAdapter;
        result.networkAdapterLongName = adapter.networkAdapterLongName;
        result.cameraIpv4Address = address;
        result.index = adapter.index;
        connectAttemptCounter = 0;
        str = "Found camera at: " + address + "";
        eventCallback(str, context, 1);
        return FOUND_CAMERA;
    }

}

void AutoConnectWindows::onFoundCamera(Result supportResult) {
    callback(result, context);
}

void AutoConnectWindows::stop() {
    loopAdapters = false;
    listenOnAdapter = false;
    shouldProgramRun = false;
    running = false;

    if (t != nullptr)
        t->join();


    delete(t); //instantiated in start func
    t = nullptr;
}

void AutoConnectWindows::run(void* instance, std::vector<Result> adapters) {
    auto *app = (AutoConnectWindows *) instance;
    app->eventCallback("Started detection service", app->context, 0);

    auto path = std::filesystem::absolute("Assets/Tools/windows/enable_jumbos.ps1").make_preferred().string();
    //access function:
           //The function returns 0 if the file has the given mode.
           //The function returns –1 if the named file does not exist or does not have the given mode
    if (access(path.c_str(), 0) == 0)
    {
        std::string startCommand = "start powershell.exe " + path;
        system(startCommand.c_str());
    }
    else
    {
        std::cout << "File not exist " << path << std::endl;
        system("pause");
    }

    // Get list of network adapters that are  supports our application
    std::string hostAddress;
    size_t i = 0;

    // Loop keeps retrying to connect on supported network adapters.
    while (app->loopAdapters) {
        
            if (i >= adapters.size()){
            i = 0;
            app->eventCallback("Tested all adapters. rerunning adapter search", app->context, 0);
            adapters = app->findEthernetAdapters(true, false);
            bool testedAllAdapters = true;
            for (const auto& a : adapters) {
                if (a.supports && !a.searched)
                    testedAllAdapters = false;
            }
            if (testedAllAdapters) {
                app->eventCallback("No other adapters found", app->context, 0);
                app->eventCallback("Finished", app->context, 0);
                break;
            }
        }
        auto adapter = adapters[i];

        if (!adapter.supports) {
            continue;
        }

        // If a camera has al ready been found on the adapter then dont re-run a search on it. Remove it from adapters list
        bool isAlreadySearched = false;
        for (const auto& found : app->ignoreAdapters) {
            if (found.index == adapter.index)
                isAlreadySearched = true;
        }
        if (isAlreadySearched) {
            adapters.erase(adapters.begin() + i);
            continue;
        }

        i++;

            std::string str = "Testing Adapter. Name: " + adapter.description;
            app->eventCallback(str, app->context, 0);

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
            if ((adhandle = pcap_open_live(adapter.networkAdapter.c_str(),    // name of the device
                                           65536,            // portion of the packet to capture.
// 65536 grants that the whole packet will be captured on all the MACs.
                                           1,                // promiscuous mode (nonzero means promiscuous)
                                           1000,            // read timeout
                                           errbuf            // error buffer
            )) == NULL) {
                fprintf(stderr, "\nUnable to open the adapter. %s is not supported by WinPcap\n", adapter.networkAdapter.c_str());
                /* Free the device list */
                return;
            }

            str = "Set adapter to listen for all activity";
            app->eventCallback(str, app->context, 0);

            str = "Waiting for packet at: " + adapter.networkAdapter;
            app->eventCallback(str, app->context, 0);
            while (app->listenOnAdapter) {

                // Timeout handler
                // Will timeout the number of MAX_CONNECTION_ATTEMPTS. After so many timeouts we retry on a new adapter
                if ((time(nullptr) - app->startTime) > TIMEOUT_INTERVAL_SECONDS &&
                    app->connectAttemptCounter < MAX_CONNECTION_ATTEMPTS) {
                    app->startTime = time(nullptr);
                    printf("\n");
                    str = "Timeout reached. Retrying... (" + std::to_string(app->connectAttemptCounter + 1) + "/" + std::to_string(MAX_CONNECTION_ATTEMPTS) + ")";
                    app->eventCallback(str, app->context, 0);
                    app->connectAttemptCounter++;
                    str = "Waiting for packet at: " + adapter.networkAdapter;
                    app->eventCallback(str, app->context, 0);
                }
                else if ((time(nullptr) - app->startTime) > TIMEOUT_INTERVAL_SECONDS &&
                    app->connectAttemptCounter >= MAX_CONNECTION_ATTEMPTS) {
                    app->startTime = time(nullptr);
                    printf("\n");
                    str = "Timeout reached. Switching to next supported adapter";
                    app->eventCallback(str, app->context, 2);
                    app->connectAttemptCounter = 0;
                    app->ignoreAdapters.push_back(adapter);
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

                char ips[255];
                //std::string ips;
                sprintf(ips, "%d.%d.%d.%d", (ih->saddr >> (8 * 0)) & 0xff,
                        (ih->saddr >> (8 * 1)) & 0xff,
                        (ih->saddr >> (8 * 2)) & 0xff,
                        (ih->saddr >> (8 * 3)) & 0xff);

                // TODO: fix this, otherwise a good reason for a crash in the future I would assume
                //char ip[15 + 1];
                //strcpy_s(ip, _countof(ip), ips.c_str());

                if (ih->protocol == 2) {

                    str = "Packet found. Source address: " + std::string(ips);
                    app->eventCallback(str, app->context, 0);

                    FoundCameraOnIp ret = app->onFoundIp(ips, adapter, 0);

                    if (ret == FOUND_CAMERA) {
                        app->ignoreAdapters.push_back(adapter);
                        app->onFoundCamera(adapter);
                        pcap_close(adhandle);
                        break;
                    } else if (ret == NO_CAMERA_RETRY) {
                        app->eventCallback("Did not find a camera. Retrying...", app->context, 2);
                        continue;
                    } else if (ret == NO_CAMERA) {
                        app->eventCallback("Did not find a camera on the adapter",app->context, 2);
                        pcap_close(adhandle);
                        break;
                    }
                }
            }
    }
}

AutoConnect::Result AutoConnectWindows::getResult()
{
    return result;
}

crl::multisense::Channel* AutoConnectWindows::getCameraChannel() {
    return cameraInterface;
}


void AutoConnectWindows::setDetectedCallback(void (*param)(Result result1, void* ctx), void* context) {
    callback = param;
    this->context = context;

}

bool AutoConnectWindows::shouldProgramClose() {
    return !shouldProgramRun;  // Note: This is just confusing usage... future technical debt right here
}

void AutoConnectWindows::setShouldProgramClose(bool close) {
    this->shouldProgramRun = !close;  // Note: This is just confusing usage... future technical debt right here
}

void AutoConnectWindows::setEventCallback(void (*param)(std::string str, void*, int)) {
    eventCallback = param;

}

void AutoConnectWindows::clearSearchedAdapters() {
    ignoreAdapters.clear();
}


void AutoConnectWindows::disableAutoConfiguration(std::string address, uint32_t index) {
    HRESULT hres;

    // Step 1: --------------------------------------------------
    // Initialize COM. ------------------------------------------

    hres = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (FAILED(hres))
    {
        std::cout << "Failed to initialize COM library. Error code = 0x"
            << hres << hres << std::endl;
        return;                  // Program has failed.
    }

    // Step 2: --------------------------------------------------
    // Set general COM security levels --------------------------

    hres = CoInitializeSecurity(
        NULL,
        -1,                          // COM authentication
        NULL,                        // Authentication services
        NULL,                        // Reserved
        RPC_C_AUTHN_LEVEL_DEFAULT,   // Default authentication 
        RPC_C_IMP_LEVEL_DELEGATE, // Default Impersonation  
        NULL,                        // Authentication info
        EOAC_NONE,                   // Additional capabilities 
        NULL                         // Reserved
    );


    if (FAILED(hres))
    {
        std::cout << "Failed to initialize security. Error code = 0x"
            << hres << std::endl;
        CoUninitialize();
        return;                    // Program has failed.
    }

    // Step 3: ---------------------------------------------------
    // Obtain the initial locator to WMI -------------------------

    IWbemLocator* pLoc = NULL;

    hres = CoCreateInstance(
        CLSID_WbemLocator,
        0,
        CLSCTX_INPROC_SERVER,
        IID_IWbemLocator, (LPVOID*)&pLoc);

    if (FAILED(hres))
    {
        std::cout << "Failed to create IWbemLocator object."
            << " Err code = 0x"
            << hres << std::endl;
        CoUninitialize();
        return;                 // Program has failed.
    }

    // Step 4: -----------------------------------------------------
    // Connect to WMI through the IWbemLocator::ConnectServer method

    IWbemServices* pSvc = NULL;

    // Connect to the root\cimv2 namespace with
    // the current user and obtain pointer pSvc
    // to make IWbemServices calls.
    hres = pLoc->ConnectServer(
        _bstr_t(L"ROOT\\CIMV2"), // Object path of WMI namespace
        NULL,                    // User name. NULL = current user
        NULL,                    // User password. NULL = current
        0,                       // Locale. NULL indicates current
        NULL,                    // Security flags.
        0,                       // Authority (for example, Kerberos)
        0,                       // Context object 
        &pSvc                    // pointer to IWbemServices proxy
    );

    if (FAILED(hres))
    {
        std::cout << "Could not connect. Error code = 0x"
            << hres << std::endl;
        pLoc->Release();
        CoUninitialize();
        return;                // Program has failed.
    }

    std::cout << "Connected to ROOT\\CIMV2 WMI namespace" << std::endl;


    // Step 5: --------------------------------------------------
    // Set security levels on the proxy -------------------------

    hres = CoSetProxyBlanket(
        pSvc,                        // Indicates the proxy to set
        RPC_C_AUTHN_WINNT,           // RPC_C_AUTHN_xxx
        RPC_C_AUTHZ_NONE,            // RPC_C_AUTHZ_xxx
        NULL,                        // Server principal name 
        RPC_C_AUTHN_LEVEL_CALL,      // RPC_C_AUTHN_LEVEL_xxx 
        RPC_C_IMP_LEVEL_DELEGATE,     // RPC_C_IMP_LEVEL_xxx
        NULL,                        // client identity
        EOAC_NONE                    // proxy capabilities 
    );

    if (FAILED(hres))
    {
        std::cout << "Could not set proxy blanket. Error code = 0x"
            << hres << std::endl;
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return;               // Program has failed.
    }

    // Step 6: --------------------------------------------------
    // Use the IWbemServices pointer to make requests of WMI ----

    // For example, get the name of the operating system
// For example, get the name of the operating system
/*IEnumWbemClassObject* pEnumerator = NULL;
hres = pSvc->ExecQuery(
    bstr_t("WQL"),
    bstr_t("SELECT * FROM Win32_NetworkAdapterConfiguration"),
    WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
    NULL,
    &pEnumerator);

if (FAILED(hres))
    {
    pSvc->Release();
    pLoc->Release();
    CoUninitialize();
    return;               // Program has failed.
    }*/

    IP_INFO ipInfo;
    INT count = 1; 
    INT fIndex = index;

    ipInfo.ip.reserve(40);
    ipInfo.netmask.reserve(40);

    ipInfo.ip.insert(0, address);
    ipInfo.netmask.insert(0, "255.255.255.0");

    // Grab class required to work on Win32_NetworkAdapterConfiguration
    IWbemClassObject* pClass = NULL;
    BSTR ClassPath = SysAllocString(L"Win32_NetworkAdapterConfiguration");
    HRESULT hr = pSvc->GetObject(ClassPath, 0, NULL, &pClass, NULL);
    SysFreeString(ClassPath);
    if (WBEM_S_NO_ERROR == hr)
    {
        // Grab pointer to the input parameter class of the method we are going to call
        BSTR MethodName_ES = SysAllocString(L"EnableStatic");
        IWbemClassObject* pInClass_ES = NULL;
        if (WBEM_S_NO_ERROR == pClass->GetMethod(MethodName_ES, 0, &pInClass_ES, NULL))
        {
            // Spawn instance of the input parameter class, so that we can stuff our parameters in
            IWbemClassObject* pInInst_ES = NULL;

            if (WBEM_S_NO_ERROR == pInClass_ES->SpawnInstance(0, &pInInst_ES))
            {
                //
                // (Step 3) - Pack desired parameters into the input class instances
                //
                // Convert from multibyte strings to wide character arrays
                wchar_t tmp_ip[40];
                SAFEARRAY* ip_list = SafeArrayCreateVector(VT_BSTR, 0, count);
                // Insert into safe arrays, allocating memory as we do so (destroying the safe array will destroy the allocated memory)
                long idx[] = { 0 };
                for (int i = 0; i < count; i++)
                {
                    mbstowcs(tmp_ip, ipInfo.ip.c_str(), 40);
                    BSTR ip = SysAllocString(tmp_ip);
                    idx[0] = i;
                    if (FAILED(SafeArrayPutElement(ip_list, idx, ip)))
                    {
                        return;
                    } // if
                    // Destroy the BSTR pointer
                    SysFreeString(ip);
                } // for

                // Convert from multibyte strings to wide character arrays
                wchar_t tmp_netmask[40];
                SAFEARRAY* netmask_list = SafeArrayCreateVector(VT_BSTR, 0, count);
                // Insert into safe arrays, allocating memory as we do so (destroying the safe array will destroy the allocated memory)
                for (int i = 0; i < count; i++)
                {
                    mbstowcs(tmp_netmask, ipInfo.netmask.c_str(), 40);
                    BSTR netmask = SysAllocString(tmp_netmask);
                    idx[0] = i;
                    if (FAILED(SafeArrayPutElement(netmask_list, idx, netmask)))
                    {
                        return;
                    } // if
                    // Destroy the BSTR pointer
                    SysFreeString(netmask);
                } // for

                // Now wrap each safe array in a VARIANT so that it can be passed to COM function
                VARIANT arg1_ES;
                VariantInit(&arg1_ES);
                arg1_ES.vt = VT_ARRAY | VT_BSTR;
                arg1_ES.parray = ip_list;

                VARIANT arg2_ES;
                VariantInit(&arg2_ES);
                arg2_ES.vt = VT_ARRAY | VT_BSTR;
                arg2_ES.parray = netmask_list;

                if ((WBEM_S_NO_ERROR == pInInst_ES->Put(L"IPAddress", 0, &arg1_ES, 0)) &&
                    (WBEM_S_NO_ERROR == pInInst_ES->Put(L"SubNetMask", 0, &arg2_ES, 0)))
                {
                    //
                    // (Step 4) - Call the methods
                    //

                    // First build the object path that specifies which network adapter we are executing a method on
                    char indexString[10];
                    itoa(fIndex, indexString, 10);

                    char instanceString[100];
                    wchar_t w_instanceString[100];
                    strcpy(instanceString, "Win32_NetworkAdapterConfiguration.Index='");
                    strcat(instanceString, indexString);
                    strcat(instanceString, "'");
                    mbstowcs(w_instanceString, instanceString, 100);
                    BSTR InstancePath = SysAllocString(w_instanceString);

                    // Now call the method
                    IWbemClassObject* pOutInst = NULL;
                    hr = pSvc->ExecMethod(InstancePath, MethodName_ES, 0, NULL, pInInst_ES, &pOutInst, NULL);
                    if (FAILED(hr))
                    {
                        // false
                        return;
                    } // if
                    SysFreeString(InstancePath);
                } // if

                // Clear the variants - does this actually get ride of safearrays?
                VariantClear(&arg1_ES);
                VariantClear(&arg2_ES);

                // Destroy safe arrays, which destroys the objects stored inside them
                //SafeArrayDestroy(ip_list); ip_list = NULL;
                //SafeArrayDestroy(netmask_list); netmask_list = NULL;
            }
            else
            {
                // false
            } // if

            // Free up the instances that we spawned
            if (pInInst_ES)
            {
                pInInst_ES->Release();
                pInInst_ES = NULL;
            } // if
        }
        else
        {
            // false
        } // if

        // Free up methods input parameters class pointers
        if (pInClass_ES)
        {
            pInClass_ES->Release();
            pInClass_ES = NULL;
        } // if
        SysFreeString(MethodName_ES);
    }
    else
    {
        // false
    } // if
    // Cleanup
    // ========

    pSvc->Release();
    pLoc->Release();
    CoUninitialize();

    return;   // Program successfully completed.
}