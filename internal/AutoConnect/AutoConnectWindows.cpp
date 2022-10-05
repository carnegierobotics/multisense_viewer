#include "AutoConnectWindows.h"
//
// Created by mgjer on 14/07/2022.
//
#ifdef _MSC_VER
/*
 * we do not want the warnings about the old deprecated and unsecure CRT functions
 * since these examples can be compiled under *nix as well
 */

#define IP_MAXPACKET 65535

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


#include "AutoConnectWindows.h"

#pragma comment(lib, "wbemuuid.lib")

#include <cstdio>
#include <fstream>
#include <iostream>
#include <Iprtrmib.h>
#include <ctime>
#include <filesystem>

#include "pcap.h"
#include "WinRegEditor.h"


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
		if (d->description)
			adapter.description = d->description;
		std::cout << d->name << std::endl;
		size_t found = std::string(d->name).find(token);
		if (found != std::string::npos)
			prefix = std::string(d->name).substr(0, found);
	}

	std::cout << "prefix: " << prefix << std::endl;

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
			if (bleFoundInName || pAdapterInfo->Type != MIB_IF_TYPE_ETHERNET) {
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
			char* con = (char*)malloc(lenA + lenB + 1);
			memcpy(con, prefix.c_str(), lenA);
			memcpy(con + lenA, pAdapterInfo->AdapterName, lenB + 1);
			adapter.networkAdapterLongName = con;
			adapter.networkAdapter = pAdapterInfo->AdapterName;
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

	WinRegEditor regEditor(adapter.networkAdapter, adapter.description, adapter.index);
	if (regEditor.ready) {
		str = "Configuring NetAdapter...";
		eventCallback(str, context, 0);
		regEditor.readAndBackupRegisty();	
		regEditor.setTCPIPValues(hostAddress, "255.255.255.0");
		regEditor.setJumboPacket("9014");
		regEditor.restartNetAdapters();
		// 8 Seconds to wait for adapter to restart. This will vary from machine to machine and should be re-done
		// If possible then wait for a windows event that triggers when the adapter is ready
		// TODO: thread_sleep - Explanation above
		std::this_thread::sleep_for(std::chrono::milliseconds(8000));
		str = "Finished Configuration";
		eventCallback(str, context, 0);
		// Wait for adapter to come back online
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
		result = adapter;
		result.cameraIpv4Address = address;
		connectAttemptCounter = 0;
		str = "Found camera at: " + address + "";
		eventCallback(str, context, 1);
		return FOUND_CAMERA;
	}

}

void AutoConnectWindows::onFoundCamera(Result supportResult) {
	callback(result, context);

	crl::multisense::Channel::Destroy(cameraInterface);
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
	auto* app = (AutoConnectWindows*)instance;
	app->eventCallback("Started detection service", app->context, 0);

	// Get list of network adapters that are  supports our application
	std::string hostAddress;
	size_t i = 0;

	// Loop keeps retrying to connect on supported network adapters.
	while (app->loopAdapters) {

		if (i >= adapters.size()) {
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

		pcap_if_t* alldevs;
		pcap_t* adhandle;
		int res;
		char errbuf[PCAP_ERRBUF_SIZE];
		struct tm* ltime;
		char timestr[16];
		struct pcap_pkthdr* header;
		const u_char* pkt_data;
		time_t local_tv_sec;

		/* Open the adapter */
		if ((adhandle = pcap_open_live(adapter.networkAdapterLongName.c_str(),    // name of the device
			65536,            // portion of the packet to capture.
			// 65536 grants that the whole packet will be captured on all the MACs.
			1,                // promiscuous mode (nonzero means promiscuous)
			1000,            // read timeout
			errbuf            // error buffer
		)) == NULL) {
			fprintf(stderr, "\nUnable to open the adapter. %s is not supported by WinPcap\n", adapter.networkAdapterLongName.c_str());
			/* Free the device list */
			return;
		}

		str = "Set adapter to listen for all activity";
		app->eventCallback(str, app->context, 0);

		str = "Waiting for packet at: " + adapter.networkAdapterLongName;
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

			// Retry if no packet was received	
			if (res  == 0)
				continue;
			/* convert the timestamp to readable format */
			local_tv_sec = header->ts.tv_sec;
			ltime = localtime(&local_tv_sec);
			strftime(timestr, sizeof timestr, "%H:%M:%S", ltime);

			/* retireve the position of the ip header */
			auto* ih = (iphdr*)(pkt_data + 14); //length of ethernet header

			char ips[255];
			//std::string ips;
			sprintf(ips, "%d.%d.%d.%d", (ih->saddr >> (8 * 0)) & 0xff,
				(ih->saddr >> (8 * 1)) & 0xff,
				(ih->saddr >> (8 * 2)) & 0xff,
				(ih->saddr >> (8 * 3)) & 0xff);

			if (ih->protocol == 2) {

				str = "Packet found. Source address: " + std::string(ips);
				app->eventCallback(str, app->context, 0);

				FoundCameraOnIp ret = app->onFoundIp(ips, adapter, 0);

				if (ret == FOUND_CAMERA) {
					app->ignoreAdapters.push_back(adapter);
					app->onFoundCamera(adapter);
					pcap_close(adhandle);
					break;
				}
				else if (ret == NO_CAMERA_RETRY) {
					app->eventCallback("Did not find a camera. Retrying...", app->context, 2);
					continue;
				}
				else if (ret == NO_CAMERA) {
					app->eventCallback("Did not find a camera on the adapter", app->context, 2);
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
