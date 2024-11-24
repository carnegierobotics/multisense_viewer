//
// Created by magnus-desktop on 11/24/24.
//

#ifndef PROJECT_H
#define PROJECT_H

namespace VkRender {
    struct Project {

        struct EditorConfig {
            int32_t borderSize = 5;
            std::string editorTypeDescription;
            int32_t width = 0;  // in pixels
            int32_t height = 0; // in pixels
            int32_t x = 0;      // x offset
            int32_t y = 0;      // y offset
        };

        std::string projectName = "MultiSense Editor";
        std::string sceneName = "Default Scene";

        std::vector<EditorConfig> editors;
    };
}

#endif //PROJECT_H
