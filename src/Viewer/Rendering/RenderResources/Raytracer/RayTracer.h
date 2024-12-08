//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RAYTRACER_H
#define MULTISENSE_VIEWER_RAYTRACER_H

#include "Viewer/Application/pch.h"
#include "Viewer/Scenes//Scene.h"
#include "Viewer/Tools/SyclDeviceSelector.h"

namespace VkRender::RT {
    class RayTracer {
    public:
        RayTracer(std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height);

        void update();


        uint8_t* getImage() {return m_imageMemory;}

        ~RayTracer();

    private:
        SyclDeviceSelector m_selector = SyclDeviceSelector(SyclDeviceSelector::DeviceType::GPU);

        std::shared_ptr<Scene> m_scene;
        uint8_t* m_imageMemory = nullptr;

        uint32_t m_width = 0, m_height = 0;

        struct GPUData {
            uint8_t* imageMemory = nullptr;

        }m_gpu;


        void saveAsPPM(const std::filesystem::path& filename) const;
    };
}


#endif //MULTISENSE_VIEWER_RAYTRACER_H
