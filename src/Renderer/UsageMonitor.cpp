//
// Created by magnus on 4/9/23.
//

#include <iostream>

#include "Viewer/Renderer/UsageMonitor.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Tools/Utils.h"

UsageMonitor::UsageMonitor() {
    auto path = Utils::getSystemCachePath() / "usage.json";
    usageFilePath = path;

    if (!std::filesystem::exists(path)) {
        initializeJSONFile();
    }
    std::ifstream input_file(path);

    // Create a string stream and copy the file's contents into it
    std::stringstream buffer;
    buffer << input_file.rdbuf();

    // Parse the JSON from the string stream
    nlohmann::json json_obj = parseJSON(buffer);

    // Output the parsed JSON object
    std::cout << json_obj.dump(4) << std::endl;
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

    std::string version = "1.1.0";
    std::ofstream output_file(usageFilePath);
    std::string jsonBoilerPlate = "{\n"
                                  "    \"version\": \"" + version + "\",\n"
                                                                    "    \"timestamp\": \"2023-04-03T15:30:00Z\",\n"
                                                                    "    \"os\": {\n"
                                                                    "        \"name\": \"Windows\",\n"
                                                                    "        \"version\": \"10.0.19042\",\n"
                                                                    "        \"architecture\": \"x86_64\"\n"
                                                                    "    },\n"
                                                                    "    \"app\": {\n"
                                                                    "        \"name\": \"MyApp\",\n"
                                                                    "        \"version\": \"1.0.0\"\n"
                                                                    "    },\n"
                                                                    "    \"stats\": {\n"
                                                                    "        \"event\": \"usage\",\n"
                                                                    "        \"data\": {\n"
                                                                    "            \"time_spent\": 1200,\n"
                                                                    "            \"files_opened\": 10,\n"
                                                                    "            \"searches_performed\": 5,\n"
                                                                    "            \"settings_changed\": {\n"
                                                                    "                \"enable_notifications\": true,\n"
                                                                    "                \"auto_save_interval\": 300\n"
                                                                    "            }\n"
                                                                    "        }\n"
                                                                    "    }\n"
                                                                    "}";

    output_file << jsonBoilerPlate;
    output_file.close();
}