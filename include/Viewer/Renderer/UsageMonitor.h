//
// Created by magnus on 4/9/23.
//

#ifndef MULTISENSE_VIEWER_USAGEMONITOR_H
#define MULTISENSE_VIEWER_USAGEMONITOR_H

#include <filesystem>
#include <json.hpp>

#include "Viewer/Core/ServerConnection.h"

class UsageMonitor {
public:
    UsageMonitor();
    void sendUsageLog();
    void setSetting(const std::string &key, const std::string &value);
    void loadSettingsFromFile();

private:
    std::filesystem::path usageFilePath;
    std::filesystem::path logFilePath;
    nlohmann::json jsonFile;
    std::unique_ptr<VkRender::ServerConnection> server;

    nlohmann::json parseJSON(const std::stringstream &buffer);

    void initializeJSONFile();

    void addEvent();

    nlohmann::json openUsageFile();
};


#endif //MULTISENSE_VIEWER_USAGEMONITOR_H
