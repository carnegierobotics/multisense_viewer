//
// Created by magnus on 4/24/24.
//

#ifndef MULTISENSE_VIEWER_APPLICATIONUSERSETTING_H
#define MULTISENSE_VIEWER_APPLICATIONUSERSETTING_H
/**
 * Settings that are changeable by the user
 */
#include <json.hpp>

#include "Viewer/Tools/Logger.h"

namespace VkRender::AppConfig {

    struct ApplicationUserSetting {
        Log::LogLevel logLevel = Log::LOG_LEVEL_INFO;
        bool sendUsageLogOnExit = true;
        /** @brief Set by user in pop up modal */
        bool userConsentToSendLogs = true;
        /** @brief If there is no prior registered consent from the user */
        bool askForUsageLoggingPermissions = true;
        std::filesystem::path lastOpenedFolderPath;
        std::filesystem::path lastOpenedImportModelFolderPath;

        std::string sceneName;
        std::string projectName;

    };

    static void to_json(nlohmann::json &j, const ApplicationUserSetting &settings) {
        j = nlohmann::json{
                {"logLevel",                        settings.logLevel},
                {"sendUsageLogOnExit",              settings.sendUsageLogOnExit},
                {"userConsentToSendLogs",           settings.userConsentToSendLogs},
                {"askForUsageLoggingPermissions",   settings.askForUsageLoggingPermissions},
                {"lastOpenedFolderPath",            settings.lastOpenedFolderPath.string()},
                {"lastOpenedImportModelFolderPath", settings.lastOpenedImportModelFolderPath.string()},
                {"sceneName",            settings.sceneName},
                {"projectName", settings.projectName},

        };
    }

    static void from_json(const nlohmann::json &j, ApplicationUserSetting &settings) {
        j.at("logLevel").get_to(settings.logLevel);
        j.at("sendUsageLogOnExit").get_to(settings.sendUsageLogOnExit);
        j.at("userConsentToSendLogs").get_to(settings.userConsentToSendLogs);
        j.at("askForUsageLoggingPermissions").get_to(settings.askForUsageLoggingPermissions);
        settings.lastOpenedFolderPath = j.at("lastOpenedFolderPath").get<std::string>();
        settings.lastOpenedImportModelFolderPath = j.at("lastOpenedImportModelFolderPath").get<std::string>();

        j.at("sceneName").get_to(settings.sceneName);
        j.at("projectName").get_to(settings.projectName);
    }
}
#endif //MULTISENSE_VIEWER_APPLICATIONUSERSETTING_H
