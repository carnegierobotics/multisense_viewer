//
// Created by magnus on 4/24/24.
//

#ifndef MULTISENSE_VIEWER_APPLICATIONUSERSETTING_H
#define MULTISENSE_VIEWER_APPLICATIONUSERSETTING_H
/**
 * Settings that are changeable by the user
 */
#include "Viewer/Tools/Logger.h"

namespace VkRender::AppConfig {

    struct ApplicationUserSettings {
        Log::LogLevel logLevel = Log::LOG_LEVEL_INFO;
        bool sendUsageLogOnExit = true;
        /** @brief Set by user in pop up modal */
        bool userConsentToSendLogs = true;
        /** @brief If there is no prior registered consent from the user */
        bool askForUsageLoggingPermissions = true;

        std::filesystem::path lastOpenedImportModelFolderPath;
        std::string projectName;
        std::filesystem::path lastActiveScenePath;

    };
}
#endif //MULTISENSE_VIEWER_APPLICATIONUSERSETTING_H
