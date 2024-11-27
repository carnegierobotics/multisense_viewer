//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RAYTRACER_H
#define MULTISENSE_VIEWER_RAYTRACER_H

#include "Viewer/Application/pch.h"
#include "Viewer/Scenes//Scene.h"

namespace VkRender::RT {
    class RayTracer {
    public:
        void setup(std::shared_ptr<Scene>& scene);

        void update(uint32_t width, uint32_t height);


        void* getImage() {return m_imageMemory;}

    private:
        std::shared_ptr<Scene> m_scene;
        void* m_imageMemory = nullptr;

        uint32_t m_width = 0, m_height = 0;
    };
}


#endif //MULTISENSE_VIEWER_RAYTRACER_H
