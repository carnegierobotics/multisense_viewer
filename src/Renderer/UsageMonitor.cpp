//
// Created by magnus on 4/9/23.
//

#include <iostream>

#include "Viewer/Renderer/UsageMonitor.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Core/RendererConfig.h"

UsageMonitor::UsageMonitor() {
    // Initialize usage file
    auto path = Utils::getSystemCachePath() / "usage.json";
    usageFilePath = path;
    logFilePath = Utils::getSystemCachePath() / "logger.log";

    if (!std::filesystem::exists(usageFilePath)) {
        initializeJSONFile();
    }
    VkRender::RendererConfig &config = VkRender::RendererConfig::getInstance();

    // Connect to CRL server
    server = std::make_unique<VkRender::ServerConnection>(config.getAnonymousIdentifier(), config.getServerInfo());
}


void UsageMonitor::loadSettingsFromFile() {
    nlohmann::json jsonObj = openUsageFile();
    VkRender::RendererConfig &config = VkRender::RendererConfig::getInstance();

    if (!jsonObj.contains("settings")) {
        nlohmann::json settingsJson;
        jsonObj["settings"] = settingsJson;
    }

    auto& setting = jsonObj["settings"];
    auto user = config.getUserSetting();

    if (setting.contains("log_level"))
        user.logLevel = Utils::getLogLevelEnumFromString(setting["log_level"]);
    if (setting.contains("send_usage_log_on_exit"))
        user.sendUsageLogOnExit = Utils::stringToBool(setting["send_usage_log_on_exit"]);

    std::string str = getSetting("user_consent_to_collect_statistics") == "true" ? "true" : "false";
    getSetting("send_usage_log_on_exit", true, str);

    user.askForUsageLoggingPermissions = shouldAskForUserConsent();
    user.userConsentToSendLogs = true;

    Log::Logger::getInstance()->info("Loaded user settings from file");
    config.setUserSetting(user);
}


void UsageMonitor::setSetting(const std::string& key, const std::string& value){
    nlohmann::json jsonObj = openUsageFile();
    if (jsonObj.contains("settings")){
        // Update the specific key in the "settings" object
        jsonObj["settings"][key] = value;
    } else {
        // Create a new "settings" object and add the key-value pair
        nlohmann::json settingsJson;
        settingsJson[key] = value;
        jsonObj["settings"] = settingsJson;
    }
    Log::Logger::getInstance()->info("User updated setting: {} to {}", key, value);

    // Save the modified JSON to the file
    std::ofstream output_file(usageFilePath); // Replace this with your usageFilePath variable
    output_file << jsonObj.dump(4);
}
std::string UsageMonitor::getSetting(const std::string& key, bool createKeyIfNotExists, const std::string& defaultValue){
    nlohmann::json jsonObj = openUsageFile();
    if (jsonObj.contains("settings")){
        if (jsonObj["settings"].contains(key)){
            Log::Logger::getInstance()->info("Get setting: {}, value: {}", key, nlohmann::to_string(jsonObj["settings"][key]).c_str());
            return jsonObj["settings"][key];
        }
    } else {
        // Create a new "settings" object and add the key-value pair
        nlohmann::json settingsJson;
        jsonObj["settings"] = settingsJson;
    }
    if (createKeyIfNotExists) {
        jsonObj["settings"][key] = defaultValue;
        Log::Logger::getInstance()->info("Fetched setting: {}, it didnt exists but was created", key);
        return jsonObj["settings"][key];
    } else {
        Log::Logger::getInstance()->info("Fetched setting: {}, but it didnt exist", key);
    }
    // Save the modified JSON to the file
    std::ofstream output_file(usageFilePath); // Replace this with your usageFilePath variable
    output_file << jsonObj.dump(4);
    return "";
}

void UsageMonitor::addEvent(){
    std::ifstream input_file(usageFilePath);
    // Create a string stream and copy the file's contents into it
    std::stringstream buffer;
    buffer << input_file.rdbuf();
}

nlohmann::json UsageMonitor::openUsageFile() {
    std::ifstream input_file(usageFilePath);
    // Create a string stream and copy the file's contents into it
    std::stringstream buffer;
    buffer << input_file.rdbuf();

    // Parse the JSON from the string stream
    nlohmann::json jsonObj = parseJSON(buffer);
    return jsonObj;
}
nlohmann::json UsageMonitor::parseJSON(const std::stringstream &buffer) {
    try {
        return nlohmann::json::parse(buffer.str());
    } catch (nlohmann::json::exception &exception) {
        Log::Logger::getInstance()->error("JSON Parse error: {}", exception.what());
        initializeJSONFile();
        Log::Logger::getInstance()->info("Reverting all usage diagnostics data and resetting json file",
                                         exception.what());

        return nullptr;
    }
}

void UsageMonitor::initializeJSONFile() {
    if (std::filesystem::exists(usageFilePath)) {
        if (!std::filesystem::remove(usageFilePath)) {
            Log::Logger::getInstance()->info("File {} deleted successfully", usageFilePath.string());
        } else {
            Log::Logger::getInstance()->error("Error deleting {}", usageFilePath.string());
        }
    }

    VkRender::RendererConfig &config = VkRender::RendererConfig::getInstance();

    std::string logVersion = "1.0.0";
    std::ofstream output_file(usageFilePath);
    std::string jsonBoilerPlate = "{\n"
                                  "    \"log_version\": \"" + logVersion + "\",\n"
                                                                        "    \"os\": {\n"
                                                                        "         \"name\": \"" + config.getOS() +
                                  "\",\n"
                                  "         \"version\": \"" + config.getOsVersion() + "\",\n"
                                                                                       "         \"architecture\": \"" +
                                  config.getArchitecture() + "\"\n"
                                                             "    },\n"
                                                             "    \"app\": {\n"
                                                             "        \"name\": \"MultiSense Viewer\",\n"
                                                             "        \"version\": \"" + config.getAppVersion() + "\"\n"
                                                                                                                  "    },\n"
                                                                                                                  "    \"stats\": {\n"
                                                                                                                  "    }\n"
                                                                                                                  "}";

    output_file << jsonBoilerPlate;
    output_file.close();
}

void UsageMonitor::sendUsageLog() {
    server->sendUsageStatistics(usageFilePath, logFilePath);
}

bool UsageMonitor::hasUserLogCollectionConsent() {
    return getSetting("user_consent_to_collect_statistics", true, "false") == "true";
}

bool UsageMonitor::shouldAskForUserConsent() {
    return getSetting("ask_user_consent_to_collect_statistics", true, "true") == "true";
}
