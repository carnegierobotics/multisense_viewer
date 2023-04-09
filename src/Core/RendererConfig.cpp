//
// Created by magnus on 4/9/23.
//

#include "Viewer/Core/RendererConfig.h"

namespace VkRender {

#ifdef WIN32
    void getOSVersion() {
                     OSVERSIONINFOEX osvi;
            ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
            osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);

            if (GetVersionEx((LPOSVERSIONINFO)&osvi)) {
                std::cout << "Windows OS version: " << osvi.dwMajorVersion << "." << osvi.dwMinorVersion << std::endl;
            } else {
                std::cerr << "Error getting Windows OS version" << std::endl;
            }
            }
#else

    std::string RendererConfig::getOSVersion() {
        // OS version
        std::ifstream ifs("/etc/os-release");
        std::string line;
        std::string version = "emtpy";
        while (std::getline(ifs, line)) {
            if (line.find("PRETTY_NAME") != std::string::npos) {
                version = line.substr(line.find("=") + 1);
                version.erase(std::remove(version.begin(), version.end(), '"'), version.end());
                break;
            }
        }
        Log::Logger::getInstance()->info("Found Ubuntu version: {}", version);
        return version;
    }

    std::string RendererConfig::getArchitecture() {
        struct utsname unameData{};
        if (uname(&unameData) != 0) {
            Log::Logger::getInstance()->error("Error calling uname()", unameData.machine);
            return "Unable to read";
        }
        Log::Logger::getInstance()->info("Found architecture: {}", unameData.machine);
        return unameData.machine;

    }

    std::string RendererConfig::getApplicationVersion() {


    }

#endif
};