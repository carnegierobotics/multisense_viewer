//
// Created by magnus on 4/9/23.
//

#ifndef MULTISENSE_VIEWER_USAGEMONITOR_H
#define MULTISENSE_VIEWER_USAGEMONITOR_H

#include <filesystem>
#include <json.hpp>

#include "Viewer/Rendering/Core/ServerConnection.h"

class UsageMonitor {
public:
    UsageMonitor();
    void sendUsageLog();
    /**
     * Get the latest version retrieved from crl server
     * @param version parameter to be filled if the app successfully fetched app version. No change if it didnt
     * @return If the app has fetched the version info successfully from crl server
     */
    bool getLatestAppVersionRemote(std::string *version);

    void userStartSession(std::chrono::system_clock::time_point startTime);
    void userEndSession();
    void userClickAction(const std::string &label, const std::string& type, const std::string &window);

private:
    std::filesystem::path usageFilePath;
    std::filesystem::path logFilePath;
    std::future<void> writeToUsageFileFuture;
    nlohmann::json jsonFile;
    std::unique_ptr<VkRender::ServerConnection> server;
    nlohmann::json parseJSON(const std::stringstream &buffer);

    void initializeJSONFile();

    void addEvent();

    nlohmann::json openUsageFile();

    std::future<bool> getAppVersionRemoteFuture;
    std::chrono::system_clock::time_point m_StartTime;
    uint32_t sessionIndex = 0;


    void saveJsonToUsageFile(nlohmann::json jsonObj);

    std::string getCurrentTimeString();

    void writeToUsageFileAsync(const nlohmann::json& obj);
};


#endif //MULTISENSE_VIEWER_USAGEMONITOR_H
