//
// Created by magnus on 4/9/23.
//

#ifndef MULTISENSE_VIEWER_USAGEMONITOR_H
#define MULTISENSE_VIEWER_USAGEMONITOR_H

#include <filesystem>
#include <json.hpp>
class UsageMonitor {
public:
    UsageMonitor();

private:
    std::filesystem::path usageFilePath;
    nlohmann::json jsonFile;


    nlohmann::json parseJSON(const std::stringstream &buffer);

    void initializeJSONFile();
};


#endif //MULTISENSE_VIEWER_USAGEMONITOR_H
