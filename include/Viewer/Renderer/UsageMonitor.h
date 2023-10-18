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
    std::string getSetting(const std::string &key, bool createIfNotExists = false, const std::string& defaultValue = "false");
    void loadSettingsFromFile();
    bool hasUserLogCollectionConsent();
    bool shouldAskForUserConsent();
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

    std::string getCurrentTimeString(std::chrono::system_clock::time_point time = std::chrono::system_clock::now());

    void writeToUsageFileAsync(const nlohmann::json& obj);
};


#endif //MULTISENSE_VIEWER_USAGEMONITOR_H
