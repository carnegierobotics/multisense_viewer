//
// Created by magnus on 4/9/23.
//

#include <iostream>
#include <httplib.h>

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
    server = std::make_unique<VkRender::ServerConnection>(config.getAnonIdentifierString(), config.getServerInfo());

}

void UsageMonitor::addEvent(){
    std::ifstream input_file(usageFilePath);
    // Create a string stream and copy the file's contents into it
    std::stringstream buffer;
    buffer << input_file.rdbuf();

    // Parse the JSON from the string stream
    nlohmann::json json_obj = parseJSON(buffer);
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
