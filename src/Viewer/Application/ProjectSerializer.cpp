//
// Created by magnus-desktop on 11/24/24.
//
#include <yaml-cpp/emitter.h>

#include "ProjectSerializer.h"

#include <yaml-cpp/yaml.h>


namespace VkRender {
    ProjectSerializer::ProjectSerializer(Project& projectInfo) : m_project(projectInfo) {
    }

    void ProjectSerializer::serialize(const std::filesystem::path& filePath) {
        auto& project = m_project;

        // Create a YAML emitter
        YAML::Emitter out;
        out << YAML::BeginMap;

        // Serialize project-level settings
        out << YAML::Key << "projectName" << YAML::Value << project.projectName;
        out << YAML::Key << "sceneName" << YAML::Value << project.sceneName;

        // Serialize editor configurations
        out << YAML::Key << "editors";
        out << YAML::Value << YAML::BeginSeq;
        for (const auto& editor : project.editors) {
            out << YAML::BeginMap;
            out << YAML::Key << "borderSize" << YAML::Value << editor.borderSize;
            out << YAML::Key << "editorTypeDescription" << YAML::Value << editor.editorTypeDescription;
            out << YAML::Key << "width" << YAML::Value << editor.width;
            out << YAML::Key << "height" << YAML::Value << editor.height;
            out << YAML::Key << "x" << YAML::Value << editor.x;
            out << YAML::Key << "y" << YAML::Value << editor.y;
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;

        out << YAML::EndMap;

        // Write to file
        std::ofstream outFile(filePath);
        if (!outFile.is_open()) {
            Log::Logger::getInstance()->error("Failed to open file for writing: {}", filePath.string());
            return;
        }
        outFile << out.c_str();
        outFile.close();

        Log::Logger::getInstance()->info("Successfully wrote editor layout to: {}", filePath.string());
    }


    void ProjectSerializer::serializeRuntime(const std::filesystem::path& filePath) {
        throw std::runtime_error("serializeRuntime Not implemented");
    }

bool ProjectSerializer::deserialize(const std::filesystem::path& filePath) const {
    std::ifstream inFile(filePath);
    if (!inFile.is_open()) {
        Log::Logger::getInstance()->error("Failed to open file for reading: {}", filePath.string());
        return false;
    }

    YAML::Node root;
    try {
        root = YAML::Load(inFile);
    } catch (const YAML::ParserException& e) {
        Log::Logger::getInstance()->error("Failed to parse YAML file: {}", e.what());
        return false;
    }

    Project project;

    // Deserialize project-level settings
    if (root["projectName"]) {
        project.projectName = root["projectName"].as<std::string>();
    }
    if (root["sceneName"]) {
        project.sceneName = root["sceneName"].as<std::string>();
    }

    // Deserialize editor configurations
    if (root["editors"] && root["editors"].IsSequence()) {
        for (const auto& editorNode : root["editors"]) {
            Project::EditorConfig editor;

            if (editorNode["borderSize"]) {
                editor.borderSize = editorNode["borderSize"].as<int32_t>();
            }
            if (editorNode["editorTypeDescription"]) {
                editor.editorTypeDescription = editorNode["editorTypeDescription"].as<std::string>();
            }
            if (editorNode["width"]) {
                editor.width = editorNode["width"].as<int32_t>();
            }
            if (editorNode["height"]) {
                editor.height = editorNode["height"].as<int32_t>();
            }
            if (editorNode["x"]) {
                editor.x = editorNode["x"].as<int32_t>();
            }
            if (editorNode["y"]) {
                editor.y = editorNode["y"].as<int32_t>();
            }

            project.editors.push_back(editor);
        }
    }

    // Update the current project
    m_project = std::move(project);
    Log::Logger::getInstance()->info("Successfully loaded project from: {}", filePath.string());
    return true;
}


    bool ProjectSerializer::deserializeRuntime(const std::filesystem::path& filePath) {
        throw std::runtime_error("deserializeRuntime Not implemented");
    }
}
