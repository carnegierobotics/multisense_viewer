//
// Created by mgjer on 03/06/2024.
//

#ifndef MULTISENSE_VIEWER_ABSTRACTRENDERER_H
#define MULTISENSE_VIEWER_ABSTRACTRENDERER_H

#include <filesystem>

#include <Viewer/Core/Camera.h>

namespace VkRender {
    class Renderer;

    class AbstractRenderer {
    public:
        struct RenderInfo {
            const Camera *camera{};
            bool debug = false;
        };

        struct InitializeInfo {
            Renderer* context = nullptr;

            const Camera *camera = nullptr;

            std::filesystem::path loadFile;

            uint32_t width, height, channels;
            uint32_t imageSize;
        };

        AbstractRenderer() = default;
        virtual ~AbstractRenderer() = default;

        virtual void setup(const InitializeInfo &initInfo) = 0;
        virtual void render(const RenderInfo &renderInfo) = 0;
        virtual uint8_t* getImage() = 0;
        virtual uint32_t getImageSize() = 0;

    };
}
#endif //MULTISENSE_VIEWER_ABSTRACTRENDERER_H
