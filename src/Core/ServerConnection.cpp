//
// Created by magnus on 4/10/23.
//

#include <filesystem>

#include "Viewer/Core/ServerConnection.h"
#include "Viewer/Tools/Logger.h"

namespace VkRender {
    void ServerConnection::sendUsageStatistics(const std::filesystem::path &usageFilePath,
                                               std::filesystem::path logFilePath) {

        // Open the file to upload
        std::ifstream jsonFileStream(usageFilePath, std::ios::binary);
        if (!jsonFileStream.is_open()) {
            Log::Logger::getInstance()->error("Failed to open usage json file: {}", usageFilePath.string());
        }
        // Open the file to upload
        std::ifstream logFileStream(logFilePath, std::ios::binary);
        if (!logFileStream.is_open()) {
            Log::Logger::getInstance()->error("Failed to open log file: {}", usageFilePath.string());
        }

        // Read the file into a buffer
        std::stringstream jsonFileBuffer;
        jsonFileBuffer << jsonFileStream.rdbuf();
        std::string jsonContent = jsonFileBuffer.str();
        // Read the file into a buffer
        std::stringstream logFileBuffer;
        logFileBuffer << logFileStream.rdbuf();
        std::string logContent = logFileBuffer.str();

        // Create the POST request with the file contents
        httplib::MultipartFormDataItems items = {
                {"user",     m_Identifier, "",              ""},
                {"userFile", jsonContent,  "userFile.json", "application/json"},
                {"logFile",  logContent,   "logFile.log",   "text/plain"}
        };

        Log::Logger::getInstance()->info("Sending usage statistics");

        auto res = m_Client->Post(m_Destination, items);
        if (res) {
            switch (res->status) {
                case 200:
                    Log::Logger::getInstance()->info("Status code 200. The server replied with: '{}'", res->body);
                    break;
                case 500:
                    Log::Logger::getInstance()->info("Status code 500. Server error: {}", res->body);
                    break;
                case 404:
                    Log::Logger::getInstance()->info("Error 404.Not found {}", res->body);
            }
        } else {
            auto err = res.error();
            Log::Logger::getInstance()->warning("Unable to service HTTP post: {} ", httplib::to_string(err));
        }
    }
};
