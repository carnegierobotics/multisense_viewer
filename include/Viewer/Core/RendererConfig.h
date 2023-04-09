//
// Created by magnus on 4/9/23.
//

#ifndef MULTISENSE_VIEWER_RENDERCONFIG_H
#define MULTISENSE_VIEWER_RENDERCONFIG_H

#include <string>
#include <fstream>
#include <sys/utsname.h>

#include "Viewer/Tools/Logger.h"

namespace VkRender {
    class RendererConfig {
    public:
        static RendererConfig &getInstance() {
            static RendererConfig instance;
            return instance;
        }


    private:
        RendererConfig() {
            // Read OS version
            m_OSVersion = getOSVersion();
            m_Architecture = getArchitecture();
        }




        std::string m_Architecture;
        std::string m_OSVersion;
        std::string m_AppVersion;
        std::string m_TimeStamp;
        std::string m_GPUDevice;


        std::string getArchitecture();
        std::string getOSVersion();
        std::string getApplicationVersion();
    };

};
#endif //MULTISENSE_VIEWER_RENDERCONFIG_H
