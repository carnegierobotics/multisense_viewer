//
// Created by mgjer on 03/06/2024.
//

#ifndef MULTISENSE_VIEWER_RAYTRACER_H
#define MULTISENSE_VIEWER_RAYTRACER_H

#include <Viewer/SYCL/AbstractRenderer.h>

namespace VkRender {


    class RayTracer : public  AbstractRenderer {
    public:
        RayTracer() = default;
        void setup(const InitializeInfo &initInfo) override;

        void render(const RenderInfo &renderInfo) override;

        ~RayTracer() override{
            if (m_image)
                free(m_image);
        }

        uint8_t* getImage() override;

    private:
        uint8_t* m_image = nullptr;
        InitializeInfo m_initInfo{};

    };

}
#endif //MULTISENSE_VIEWER_RAYTRACER_H
