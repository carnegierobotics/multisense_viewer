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

    struct EditorUIState {
        // Define specific UI settings here
        bool enableSecondaryView = false;
        bool fixAspectRatio = false;
    };

    struct ApplicationUserSetting {
        Log::LogLevel logLevel = Log::LOG_LEVEL_INFO;
        bool sendUsageLogOnExit = true;
        /** @brief Set by user in pop up modal */
        bool userConsentToSendLogs = true;
        /** @brief If there is no prior registered consent from the user */
        bool askForUsageLoggingPermissions = true;

        /** @brief UI application lifetime settings **/
        EditorUIState editorUiState;

        std::filesystem::path lastOpenedFolderPath;
        std::filesystem::path lastOpenedImportModelFolderPath;

        uint32_t applicationWidth = 1480;
        uint32_t applicationHeight = 720;
    };

    static void to_json(nlohmann::json &j, const ApplicationUserSetting &settings) {
        j = nlohmann::json{
                {"logLevel",                        settings.logLevel},
                {"sendUsageLogOnExit",              settings.sendUsageLogOnExit},
                {"userConsentToSendLogs",           settings.userConsentToSendLogs},
                {"askForUsageLoggingPermissions",   settings.askForUsageLoggingPermissions},
                {"lastOpenedFolderPath",            settings.lastOpenedFolderPath.string()},
                {"enableSecondaryView",             settings.editorUiState.enableSecondaryView},
                {"fixAspectRatio",                  settings.editorUiState.fixAspectRatio},
                {"lastOpenedImportModelFolderPath", settings.lastOpenedImportModelFolderPath.string()},
                {"applicationWidth",                settings.applicationWidth},
                {"applicationHeight",               settings.applicationHeight},

        };
    }

    static void from_json(const nlohmann::json &j, ApplicationUserSetting &settings) {
        j.at("logLevel").get_to(settings.logLevel);
        j.at("sendUsageLogOnExit").get_to(settings.sendUsageLogOnExit);
        j.at("userConsentToSendLogs").get_to(settings.userConsentToSendLogs);
        j.at("askForUsageLoggingPermissions").get_to(settings.askForUsageLoggingPermissions);
        j.at("enableSecondaryView").get_to(settings.editorUiState.enableSecondaryView);
        j.at("fixAspectRatio").get_to(settings.editorUiState.fixAspectRatio);
        settings.lastOpenedFolderPath = j.at("lastOpenedFolderPath").get<std::string>();
        settings.lastOpenedImportModelFolderPath = j.at("lastOpenedImportModelFolderPath").get<std::string>();
        j.at("applicationWidth").get_to(settings.applicationWidth);
        j.at("applicationHeight").get_to(settings.applicationHeight);
    }
}
#endif //MULTISENSE_VIEWER_APPLICATIONUSERSETTING_H
