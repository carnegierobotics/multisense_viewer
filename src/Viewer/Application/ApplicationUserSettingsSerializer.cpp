//
// Created by magnus-desktop on 10/17/24.
//

#include "ApplicationUserSettingsSerializer.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>

namespace VkRender {



    ApplicationUserSettingsSerializer::ApplicationUserSettingsSerializer(AppConfig::ApplicationUserSettings & settings)
            : m_settings(settings) {}

    // Helper function to convert settings to YAML
    static void to_yaml(YAML::Emitter &out, const AppConfig::ApplicationUserSettings &settings) {
        out << YAML::BeginMap;
        out << YAML::Key << "logLevel" << YAML::Value << Log::logLevelToString(settings.logLevel);
        out << YAML::Key << "sendUsageLogOnExit" << YAML::Value << settings.sendUsageLogOnExit;
        out << YAML::Key << "userConsentToSendLogs" << YAML::Value << settings.userConsentToSendLogs;
        out << YAML::Key << "askForUsageLoggingPermissions" << YAML::Value << settings.askForUsageLoggingPermissions;
        out << YAML::Key << "lastOpenedImportModelFolderPath" << YAML::Value << settings.lastOpenedImportModelFolderPath.string();
        out << YAML::Key << "projectName" << YAML::Value << settings.projectName;
        out << YAML::EndMap;
    }

    // Helper function to load settings from YAML
    static void from_yaml(const YAML::Node &node, AppConfig::ApplicationUserSettings &settings) {
        settings.logLevel = Log::logLevelFromString(node["logLevel"].as<std::string>());
        settings.sendUsageLogOnExit = node["sendUsageLogOnExit"].as<bool>();
        settings.userConsentToSendLogs = node["userConsentToSendLogs"].as<bool>();
        settings.askForUsageLoggingPermissions = node["askForUsageLoggingPermissions"].as<bool>();
        settings.lastOpenedImportModelFolderPath = node["lastOpenedImportModelFolderPath"].as<std::string>();
        settings.projectName = node["projectName"].as<std::string>();
    }

    // Serialize settings to YAML
    void ApplicationUserSettingsSerializer::serialize(const std::filesystem::path& filePath) {
        YAML::Emitter out;
        to_yaml(out, m_settings);
        std::ofstream fout(filePath);
        if (fout.is_open()) {
            fout << out.c_str();
            fout.close();
        } else {
            Log::Logger::getInstance()->warning("Failed to open file for writing at: {} ", filePath.string());
        }
    }

    // Serialize runtime settings (you can customize this if runtime settings differ)
    void ApplicationUserSettingsSerializer::serializeRuntime(const std::filesystem::path& filePath) {
        serialize(filePath); // Assuming runtime settings are the same for now
    }

    // Deserialize settings from YAML
    bool ApplicationUserSettingsSerializer::deserialize(const std::filesystem::path& filePath) const
    {
        try {
            YAML::Node config = YAML::LoadFile(filePath.string());
            from_yaml(config, m_settings);
            return true;
        } catch (const std::exception &e) {
            Log::Logger::getInstance()->warning("Failed to load YAML file at: {} ", filePath.string());

            return false;
        }
    }

    // Deserialize runtime settings (can customize this for runtime-specific settings)
    bool ApplicationUserSettingsSerializer::deserializeRuntime(const std::filesystem::path& filePath) {
        return deserialize(filePath); // Assuming runtime settings are the same for now
    }

}