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
        out << YAML::Key << "LogLevel" << YAML::Value << Log::logLevelToString(settings.logLevel);
        out << YAML::Key << "SendUsageLogOnExit" << YAML::Value << settings.sendUsageLogOnExit;
        out << YAML::Key << "UserConsentToSendLogs" << YAML::Value << settings.userConsentToSendLogs;
        out << YAML::Key << "AskForUsageLoggingPermissions" << YAML::Value << settings.askForUsageLoggingPermissions;
        out << YAML::Key << "LastOpenedImportModelFolderPath" << YAML::Value << settings.lastOpenedImportModelFolderPath.string();
        out << YAML::Key << "ProjectName" << YAML::Value << settings.projectName;
        out << YAML::Key << "LastActiveScenePath" << YAML::Value << settings.lastActiveScenePath.string();
        out << YAML::Key << "AssetsPath" << YAML::Value << settings.assetsPath.string();
        out << YAML::EndMap;
    }

    // Helper function to load settings from YAML
    static void from_yaml(const YAML::Node &node, AppConfig::ApplicationUserSettings &settings) {
        settings.logLevel = Log::logLevelFromString(node["LogLevel"].as<std::string>());
        settings.sendUsageLogOnExit = node["SendUsageLogOnExit"].as<bool>();
        settings.userConsentToSendLogs = node["UserConsentToSendLogs"].as<bool>();
        settings.askForUsageLoggingPermissions = node["AskForUsageLoggingPermissions"].as<bool>();
        settings.lastOpenedImportModelFolderPath = node["LastOpenedImportModelFolderPath"].as<std::string>();
        settings.projectName = node["ProjectName"].as<std::string>();
        settings.lastActiveScenePath = node["LastActiveScenePath"].as<std::string>();
        settings.assetsPath = node["AssetsPath"].as<std::string>();
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