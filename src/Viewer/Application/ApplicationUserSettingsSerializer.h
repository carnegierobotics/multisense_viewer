//
// Created by magnus-desktop on 10/17/24.
//

#ifndef APPLICATIONUSERSETTINGSERIALIZER_H
#define APPLICATIONUSERSETTINGSERIALIZER_H

#include <filesystem>

#include "ApplicationUserSettings.h"


namespace VkRender {
    class ApplicationUserSettingsSerializer {

    public:
        explicit ApplicationUserSettingsSerializer(AppConfig::ApplicationUserSettings& settings);

        void serialize(const std::filesystem::path& filePath);
        void serializeRuntime(const std::filesystem::path& filePath);

        bool deserialize(const std::filesystem::path& filePath) const;
        bool deserializeRuntime(const std::filesystem::path& filePath);

    private:
        AppConfig::ApplicationUserSettings& m_settings;
    };
}


#endif //APPLICATIONUSERSETTINGSERIALIZER_H
