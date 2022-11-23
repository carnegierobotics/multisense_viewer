#pragma once

#include <windows.h>
#include <WinReg.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <fstream>
#include <iphlpapi.h>

#pragma comment(lib,"Advapi32.lib")

#include "../../external/simpleini/SimpleIni.h"

// Needed parameters for this winreg to configure static IP address:
// 1. Interface UUID name to: Disable DHCP, Set IP address and subnet mask	: To modify the correct adapter
// 2. IP Address and Subnet mask											: To sync camera and pc to same network
// 3. Interface Description (Lenovo USB Ethernet)							: To set MTU/Enable Jumbo
// 4. Interface Index to													: To restart adapter effectively applying changes

// Needed parameters for this winreg to configure temporary IP address:
//  The IPv4 address exists only as long as the adapter object exists. Restarting the computer destroys the IPv4 address, as does manually resetting the network interface card (NIC).
// To create non-persistent ipv4 address pass into the constructor:
// 1. adapter index
// 2. IP address to configure
// 3. SubNet Mask
class WinRegEditor {


public:
	HKEY tcpIpKey;
	HKEY adapterKey;
	uint32_t index;
	bool ready = false;
	std::string name;
	std::string adapterDesc;
	HKEY startupKeyRes{};

	// non-persistent IP
	ULONG NTEContext = 0;
	ULONG NTEInstance = 0;

	WinRegEditor(uint32_t ifIndex, std::string ipv4Addr, std::string subnetMask) {

		setStaticIp(ifIndex, ipv4Addr, subnetMask);
	}

	bool setStaticIp(uint32_t ifIndex, std::string ipv4Addr, std::string subnetMask) {
		UINT  iaIPAddress = inet_addr(ipv4Addr.c_str());
		UINT iaIPMask = inet_addr(subnetMask.c_str());

		DWORD dwRetVal = AddIPAddress(iaIPAddress, iaIPMask, ifIndex,
			&NTEContext, &NTEInstance);
		if (dwRetVal != NO_ERROR) {
			printf("AddIPAddress call failed with %d\n", dwRetVal);
			return false;
		}

		return true;
	}

	bool deleteStaticIP() {
		DWORD dwRetVal = DeleteIPAddress(NTEContext);
		if (dwRetVal != NO_ERROR) {
			printf("\tDeleteIPAddress failed with error: %d\n", dwRetVal);
		}
	}

	/**@brief
	@param lpkey: suggestion {7a71db7f-b10a-4fa2-8493-30ad4e2a947d}
	@param adapterDescription: Lenovo USB Ethernet
	**/
	WinRegEditor(std::string lpKey, std::string adapterDescription, uint32_t index) {
		// {7a71db7f-b10a-4fa2-8493-30ad4e2a947d}
		this->name = lpKey;
		this->index = index;
		this->adapterDesc = adapterDescription;

		DWORD lResult = RegOpenKeyEx(HKEY_LOCAL_MACHINE, ("SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters\\Interfaces\\" + lpKey).c_str(), 0, KEY_READ | KEY_SET_VALUE, &tcpIpKey);
		adapterKey = findAdapterKey(adapterDescription);
		if (adapterKey == nullptr) {
			std::cout << "Failed to retrieve adapter key\n";
		}
		else {
			ready = true;
		}

		long res = RegCreateKeyA(
			HKEY_LOCAL_MACHINE,
			"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunOnce",
			&startupKeyRes
		);

		if (res != ERROR_SUCCESS)
			printf("Something went wrong creating RunOnce Key\n");

		std::string cmd = "\"C:\\Program Files (x86)\\MultiSense Viewer\\Assets\\Tools\\windows\\RegistryBackup.exe\"";

		res = RegSetKeyValueA(
			startupKeyRes,
			NULL,
			"RevertSettings",
			REG_SZ,
			cmd.c_str(),
			cmd.size()
		);

	}

	struct PreChange {
		DWORD EnableDHCP = 1;
		std::string IPAddress = "";
		std::string SubnetMask = "";
		std::string JumboPacket = "";
	}backup;

	void dontLaunchOnReboot() {
		LSTATUS res = RegDeleteKeyValueA(
			startupKeyRes,
			NULL,
			"RevertSettings"
		);

		if (res != ERROR_SUCCESS) {
			printf("Failed to delete key: revert settings on reboot\n");
		}

		RegCloseKey(startupKeyRes);
	}

	void restartNetAdapters() {
		std::string strIdx = std::to_string(index);
		auto shl = ShellExecuteA(0, 0, "powershell.exe", std::string("Get-NetAdapter -InterfaceIndex " + strIdx + " | Restart-NetAdapter").c_str(), "", SW_HIDE);
	}

	int revertSettings() {
		if (!parseBackupIni())
			return -1;

		DWORD ret = -1;
		ret = RegSetValueExA(tcpIpKey, "EnableDHCP", 0, REG_DWORD, (const BYTE*)&backup.EnableDHCP, sizeof(DWORD));
		if (ret != 0)
		{
			std::cout << "Failed to reset EnableDHCP\n";
			return 1;
		}

		ret = RegSetValueExA(tcpIpKey, "IPAddress", 0, REG_MULTI_SZ, (const BYTE*)backup.IPAddress.c_str(), backup.IPAddress.size() + 1);

		if (ret != 0) {
			std::cout << "Failed to reset IPADDRESS\n";
			return 1;
		}

		ret = RegSetValueExA(tcpIpKey, "SubnetMask", 0, REG_MULTI_SZ, (const BYTE*)backup.SubnetMask.c_str(), backup.SubnetMask.size() + 1);

		if (ret != 0) {
			std::cout << "Failed to reset SubnetMask\n";
			return 1;
		}
	}

	void readAndBackupRegisty() {
		std::vector<void*> data(255);
        DWORD size = 255;
		DWORD ret;
		u_long dwType;

		ret = RegGetValueA(tcpIpKey, NULL, "IPAddress", RRF_RT_REG_MULTI_SZ, &dwType, data.data(), &size);
		if (ret != ERROR_SUCCESS) {
			std::cout << "Error, Reading IPAddress" << std::endl;
		}
		backup.IPAddress.reserve(size);
		memcpy(backup.IPAddress.data(), data.data(), size);
		size = 255;

		ret = RegGetValueA(tcpIpKey, NULL, "SubnetMask", RRF_RT_REG_MULTI_SZ, &dwType, data.data(), &size);
		if (ret != ERROR_SUCCESS) {
			std::cout << "Error, Reading SubnetMask" << std::endl;
		}
		backup.SubnetMask.reserve(size);
		memcpy(backup.SubnetMask.data(), data.data(), size);
		size = 255;
		ret = RegGetValueA(tcpIpKey, NULL, "EnableDHCP", RRF_RT_REG_DWORD, &dwType, data.data(), &size);
		if (ret != ERROR_SUCCESS) {
			std::cout << "Error, Reading EnableDHCP" << std::endl;
		}
		memcpy(&backup.EnableDHCP, data.data(), size);
		size = 255;
		ret = RegGetValueA(adapterKey, NULL, "*JumboPacket", RRF_RT_REG_SZ, &dwType, data.data(), &size);
		if (ret != ERROR_SUCCESS) {
			std::cout << "Error, Reading JumboPacket" << std::endl;
		}
		backup.JumboPacket.reserve(size);
		memcpy(backup.JumboPacket.data(), data.data(), size);

		// Write to ini file.
		CSimpleIniA ini;
		ini.SetUnicode();
		SI_Error rc = ini.LoadFile("winreg.ini");
		if (rc < 0) {
			// File doesn't exist error, then create one
			if (rc == SI_FILE && errno == ENOENT) {
				std::ofstream output = std::ofstream("winreg.ini");
				rc = ini.LoadFile("winreg.ini");
			}
		}
		if (!ini.SectionExists(name.c_str())) {
			ret = ini.SetValue(name.c_str(), "Description", adapterDesc.c_str());
			ret = ini.SetValue(name.c_str(), "Index", std::to_string(index).c_str());
			ret = ini.SetValue(name.c_str(), "Key", name.c_str());

			ret = ini.SetValue(name.c_str(), "IPAddress", backup.IPAddress.c_str());
			ret = ini.SetValue(name.c_str(), "SubnetMask", backup.SubnetMask.c_str());
			ret = ini.SetValue(name.c_str(), "EnableDHCP", std::to_string(backup.EnableDHCP).c_str());
			ret = ini.SetValue(name.c_str(), "*JumboPacket", backup.JumboPacket.c_str());
			rc = ini.SaveFile("winreg.ini");
		}
		if (rc < 0) {
		}

	}

	int setTCPIPValues(std::string ip, std::string subnetMask) {
		// Disable DHCP
		DWORD var = 0;
		DWORD ret = RegSetValueExA(tcpIpKey, "EnableDHCP", 0, REG_DWORD, (const BYTE*)&var, sizeof(DWORD));
		if (ret != ERROR_SUCCESS)
		{
			std::cout << "Failed to Set EnableDHCP to false\n";
			return 1;
		}
		// IPAddress
		//ip.push_back((char) "\0");
		ret = RegSetValueExA(tcpIpKey, "IPAddress", 0, REG_MULTI_SZ, (const BYTE*)ip.c_str(), ip.size());
		if (ret != ERROR_SUCCESS) {
			std::cout << "Failed to Set IPADDRESS\n";
			return 1;
		}
		// SubnetMask
		//subnetMask.push_back((char) "\0");
		ret = RegSetValueExA(tcpIpKey, "SubnetMask", 0, REG_MULTI_SZ, (const BYTE*)subnetMask.c_str(), subnetMask.size());
		if (ret != ERROR_SUCCESS) {
			std::cout << "Failed to Set SubnetMask\n";
			return 1;
		}
	}

	int setJumboPacket(std::string value) {
		// Set *JumboPacket
		DWORD ret = RegSetValueExA(adapterKey, "*JumboPacket", 0, REG_SZ, (const BYTE*)value.c_str(), value.size());
		if (ret != ERROR_SUCCESS) {
			std::cout << "Failed to Set *JumboPacket\n";
			return -1;
		}
		else {
			std::cout << "Set *JumboPacket on device: " << "Lenovo USB Ethernet" << std::endl;
		}
	}

	int resetJumbo() {
		if (!parseBackupIni())
			return -1;

		return setJumboPacket(backup.JumboPacket);
	}

	bool parseBackupIni() {
		CSimpleIniA ini;
		ini.SetUnicode();
		SI_Error rc = ini.LoadFile("winreg.ini");
		if (rc < 0) {
			std::cerr << "Backup register file does not exist" << std::endl;
			return false;
		}
		if (!ini.SectionExists(name.c_str())) {
			std::cerr << "Backup for adapqter does not exists " << name.c_str() << std::endl;
			return false;
		}
		backup.IPAddress = ini.GetValue(name.c_str(), "IPAddress", "");
		backup.SubnetMask = ini.GetValue(name.c_str(), "SubnetMask", "");
		backup.EnableDHCP = std::stoi(ini.GetValue(name.c_str(), "EnableDHCP", "1"));
		backup.JumboPacket = ini.GetValue(name.c_str(), "*JumboPacket", "");
		return true;
	}

private:
	HKEY findAdapterKey(std::string driverDesc) {
		HKEY queryKey;
		DWORD lResult2 = RegOpenKeyEx(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e972-e325-11ce-bfc1-08002be10318}", 0, KEY_READ, &queryKey);
		// Open the GUID class 4d36e972-e325-11ce-bfc1-08002be10318
		// List the sub keys in this category. I will be looking for devices with iftype = 6
		// Further distinguish devices by looking at the:
		// - DriverDesc
		// - NetCfgInstanceid (hex string) 
		// Once found then set the *JumboPacket value

		const DWORD MAX_KEY_LENGTH = 255;
		const DWORD MAX_VALUE_NAME = 255;

		CHAR* achClass = new char[200];
		DWORD    cbName = 0;                   // size of name string 
		DWORD    cchClassName = 0;  // size of class string 
		DWORD    cSubKeys = 0;               // number of subkeys 
		DWORD    cbMaxSubKey = 0;              // longest subkey size 
		DWORD    cchMaxClass = 0;              // longest class string 
		DWORD    cValues = 0;              // number of values for key 
		DWORD    cchMaxValue = 0;          // longest value name 
		DWORD    cbMaxValueData = 0;       // longest value data 
		DWORD    cbSecurityDescriptor = 0; // size of security descriptor 
		PFILETIME  ftLastWriteTime{};      // last write time 

		DWORD retCode;

		// Get the class name and the value count. 
		retCode = RegQueryInfoKeyA(
			queryKey,                    // key handle 
			achClass,                // buffer for class name 
			&cchClassName,           // size of class string 
			NULL,                    // reserved 
			&cSubKeys,               // number of subkeys 
			&cbMaxSubKey,            // longest subkey size 
			&cchMaxClass,            // longest class string 
			&cValues,                // number of values for this key 
			&cchMaxValue,            // longest value name 
			&cbMaxValueData,         // longest value data 
			&cbSecurityDescriptor,   // security descriptor 
			NULL);       // last write time 
		TCHAR    achKey[MAX_KEY_LENGTH];   // buffer for subkey name




		if (cSubKeys)
		{
			for (int i = 0; i < cSubKeys; i++)
			{
				cbName = MAX_KEY_LENGTH;
				retCode = RegEnumKeyEx(queryKey, i,
					achKey,
					&cbName,
					NULL,
					NULL,
					NULL,
					NULL);
				if (retCode == ERROR_SUCCESS)
				{
					printf(TEXT("(%d) %s\n"), i + 1, achKey);
					// For each subkey
					HKEY subHKey{};

					DWORD dwType = 0;
					DWORD size = 256;
					std::vector<void*> data(256);

					DWORD ret = RegGetValueA(queryKey, achKey, "DriverDesc", RRF_RT_ANY, &dwType, data.data(), &size);
					if (ret != 0) {
						std::cout << "Failed to get DriverDesc\n";
					}
					else {
						std::string description((const char*)data.data());
						if (description == driverDesc) {
							HKEY hKey;

							// Open new Key here
							std::string adapterlpKey = "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e972-e325-11ce-bfc1-08002be10318}\\" + std::string(achKey);
							DWORD lResult = RegOpenKeyEx(HKEY_LOCAL_MACHINE, adapterlpKey.c_str(), 0, KEY_READ | KEY_SET_VALUE, &hKey);
							if (lResult != ERROR_SUCCESS) {
								printf("Failed to retrieve the adapter key\n");
							}
							RegCloseKey(queryKey);
							printf("Got the adapter key\n");
							return hKey;

						}
					}
				}
			}
		}
		RegCloseKey(queryKey);
		return nullptr;
	}


};