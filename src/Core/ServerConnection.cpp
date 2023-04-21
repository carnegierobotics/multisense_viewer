//
// Created by magnus on 4/10/23.
//

#include <filesystem>

#include "Viewer/Core/ServerConnection.h"
#include "Viewer/Tools/Logger.h"

namespace VkRender {
    void ServerConnection::sendUsageStatistics(std::filesystem::path usageFilePath) {

        // Open the file to upload
        std::ifstream file_stream(usageFilePath, std::ios::binary);
        if (!file_stream.is_open()) {
            std::cerr << "Failed to open file" << std::endl;
        }

        // Read the file into a buffer
        std::stringstream file_buffer;
        file_buffer << file_stream.rdbuf();
        std::string file_contents = file_buffer.str();

        // Create the POST request with the file contents
        httplib::MultipartFormDataItems items = {
                {"user", m_Identifier,  "",           ""},
                {"file", file_contents, "usage.json", "application/json"}
        };

        Log::Logger::getInstance()->info("Sending usage statistics");

        auto res = m_Client->Post(m_Destination, items);
        if (res) {
            if (res->status == 200) {
                Log::Logger::getInstance()->info("The server replied with {}", res->body);

            }
        } else {
            auto err = res.error();
            Log::Logger::getInstance()->warning("Unable to service HTTP post: {} ", httplib::to_string(err));
        }
    }
};
