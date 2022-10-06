#pragma once

#include <windows.h>
#include <WinReg.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <fstream>

#pragma comment(lib,"Advapi32.lib")

#include <MultiSense/external/simpleini/SimpleIni.h>

// Needed parameters for this program:
// 1. Interface UUID name to: Disable DHCP, Set IP address and subnet mask	: To modify the correct adapter
// 2. IP Address and Subnet mask											: To sync camera and pc to same network
// 3. Interface Description (Lenovo USB Ethernet)							: To set MTU/Enable Jumbo
// 4. Interface Index to													: To restart adapter effectively applying changes


class WinRegEditor {

	/**@brief
	@param lpkey: suggestion {7a71db7f-b10a-4fa2-8493-30ad4e2a947d}
	@param adapterDescription: Lenovo USB Ethernet
	**/
public:
	HKEY tcpIpKey;
	HKEY adapterKey;
	uint32_t index;
	bool ready = false;
	std::string name;

	WinRegEditor(std::string lpKey, std::string adapterDescription, uint32_t index) {
		// {7a71db7f-b10a-4fa2-8493-30ad4e2a947d}
		this->name = lpKey;
		this->index = index;

		DWORD lResult = RegOpenKeyEx(HKEY_LOCAL_MACHINE, ("SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters\\Interfaces\\" + lpKey).c_str(), 0, KEY_READ | KEY_SET_VALUE, &tcpIpKey);
		adapterKey = findAdapterKey(adapterDescription);
		if (adapterKey == nullptr) {
			std::cout << "Failed to retrieve adapter key\n";
		}
		else {
			ready = true;
		}
	}

	struct PreChange {
		DWORD EnableDHCP = 1;
		std::string IPAddress = "";
		std::string SubnetMask = "";
		std::string JumboPacket = "";
	}backup;

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
		VOID* data = malloc(255);
		DWORD size = 255;
		DWORD ret;
		u_long dwType;

		ret = RegGetValueA(tcpIpKey, NULL, "IPAddress", RRF_RT_REG_MULTI_SZ, &dwType, data, &size);
		if (ret != ERROR_SUCCESS) {
			std::cout << "Error, Reading IPAddress" << std::endl;
		}
		backup.IPAddress.reserve(size);
		memcpy(backup.IPAddress.data(), data, size);
		size = 255;

		ret = RegGetValueA(tcpIpKey, NULL, "SubnetMask", RRF_RT_REG_MULTI_SZ, &dwType, data, &size);
		if (ret != ERROR_SUCCESS) {
			std::cout << "Error, Reading SubnetMask" << std::endl;
		}
		backup.SubnetMask.reserve(size);
		memcpy(backup.SubnetMask.data(), data, size);
		size = 255;
		ret = RegGetValueA(tcpIpKey, NULL, "EnableDHCP", RRF_RT_REG_DWORD, &dwType, data, &size);
		if (ret != ERROR_SUCCESS) {
			std::cout << "Error, Reading EnableDHCP" << std::endl;
		}
		memcpy(&backup.EnableDHCP, data, size);
		size = 255;
		ret = RegGetValueA(adapterKey, NULL, "*JumboPacket", RRF_RT_REG_SZ, &dwType, data, &size);
		if (ret != ERROR_SUCCESS) {
			std::cout << "Error, Reading JumboPacket" << std::endl;
		}
		backup.JumboPacket.reserve(size);
		memcpy(backup.JumboPacket.data(), data, size);
		free(data);

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

		backup.JumboPacket.push_back((char)"\0");
		if (ret != ERROR_SUCCESS) {
			std::cout << "Failed to Set *JumboPacket\n";
			return -1;
		}
		else {
			std::cout << "Set *JumboPacket on device: " << "Lenovo USB Ethernet" << std::endl;
		}
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
		HKEY hKey;
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
					HKEY subHKey;

					DWORD dwType = 0;
					DWORD size = 256;
					VOID* data = malloc(256);


					DWORD ret = RegGetValueA(queryKey, achKey, "DriverDesc", RRF_RT_ANY, &dwType, data, &size);
					if (ret != 0) {
						std::cout << "Failed to get DriverDesc\n";
					}
					else {
						std::string description((const char*)data);
						if (description == driverDesc) {

							// Open new Key here
							std::string adapterlpKey = "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e972-e325-11ce-bfc1-08002be10318}\\" + std::string(achKey);
							LPCSTR lpKeyName = adapterlpKey.c_str();
							DWORD lResult = RegOpenKeyEx(HKEY_LOCAL_MACHINE, lpKeyName, 0, KEY_READ | KEY_SET_VALUE, &hKey);

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