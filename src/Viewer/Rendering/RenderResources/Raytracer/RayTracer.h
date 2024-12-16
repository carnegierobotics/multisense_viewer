//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RAYTRACER_H
#define MULTISENSE_VIEWER_RAYTRACER_H

#include <Viewer/Rendering/MeshManager.h>

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Tools/SyclDeviceSelector.h"
#include "Viewer/Rendering/RenderResources/Raytracer/Definitions.h"

namespace VkRender::RT {
    class RayTracer {
    public:
        RayTracer(Application* context, std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height);

        void update(bool update);


        uint8_t* getImage() {return m_imageMemory;}

        ~RayTracer();


    private:
        BaseCamera m_camera;
        Application* m_context;
        SyclDeviceSelector m_selector = SyclDeviceSelector(SyclDeviceSelector::DeviceType::GPU);

        std::shared_ptr<Scene> m_scene;
        uint8_t* m_imageMemory = nullptr;

        uint32_t m_width = 0, m_height = 0;

        GPUData m_gpu;
        MeshManager m_meshManager;


        void saveAsPPM(const std::filesystem::path& filename) const;
    };
}


#endif //MULTISENSE_VIEWER_RAYTRACER_H
