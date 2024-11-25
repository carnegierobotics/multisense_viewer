//
// Created by magnus-desktop on 11/24/24.
//

#ifndef EDITORLAYOUTSERIALIZER_H
#define EDITORLAYOUTSERIALIZER_H

#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/Application/Project.h"


namespace VkRender {

class ProjectSerializer {


public:
    explicit ProjectSerializer(Project& createInfo);

    void serialize(const std::filesystem::path& filePath);
    void serializeRuntime(const std::filesystem::path& filePath);

    bool deserialize(const std::filesystem::path& filePath) const;
    bool deserializeRuntime(const std::filesystem::path& filePath);

private:
    Project& m_project;
};
}



#endif //EDITORLAYOUTSERIALIZER_H
