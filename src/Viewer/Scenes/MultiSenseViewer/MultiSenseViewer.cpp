//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/VkRender/Renderer.h"

#include "Viewer/Scenes/MultiSenseViewer/MultiSenseViewer.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"

namespace VkRender {

    std::vector<ColmapCameraPose> MultiSenseViewer::loadColmapCameras(const std::string &filePath) {
        std::ifstream file(filePath);
        std::vector<ColmapCameraPose> cameraPoses;
        std::string line;

        while (std::getline(file, line)) {
            // Skip comment lines
            if (line[0] == '#') continue;

            ColmapCameraPose pose;

            int imageId, cameraId;
            float qw, qx, qy, qz, tx, ty, tz;

            // Parse the first line of the image entry
            std::istringstream iss(line);
            iss >> imageId >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cameraId >> pose.imageName;

            // Convert quaternion from COLMAP (qw, qx, qy, qz) to glm::quat
            pose.rotation = glm::quat(qw, qx, qy, qz);

            // Translation vector (tx, ty, tz)
            pose.translation = glm::vec3(tx, ty, tz);

            cameraPoses.push_back(pose);

            // Skip the second line of the image entry (the 2D-3D correspondences)
            std::getline(file, line);
        }

        return cameraPoses;
    }
    void MultiSenseViewer::applyColmapCameraPoses(const std::vector<ColmapCameraPose> &cameraPoses) {
        for (const auto &pose : cameraPoses) {
            // Create or find your camera entity
            auto cameraEntity = createNewCamera(pose.imageName, 400, 400);
            auto &cameraComponent = cameraEntity.getComponent<CameraComponent>();

            // Set the camera's transform based on the COLMAP pose
            auto &transform = cameraEntity.getComponent<TransformComponent>();

            // Set translation (position)
            transform.setPosition(pose.translation);

            // Set rotation using the quaternion from COLMAP
            transform.setQuaternion(pose.rotation);

            // Update camera component pose
            cameraComponent.camera.pose.pos = transform.getPosition();
            cameraComponent.camera.pose.q = transform.getQuaternion();

            // Optionally add other components or handle specific camera configurations here
            cameraEntity.addComponent<MeshComponent>(1);
        }
    }


    MultiSenseViewer::MultiSenseViewer(Renderer &ctx) {
        m_sceneName = "MultiSense Viewer";

        {
            auto entity = createEntity("S30Camera");
            auto &modelComponent = entity.addComponent<MeshComponent>(Utils::getModelsPath() / "obj" / "s30.obj");
            //auto &modelComponent = entity.addComponent<MeshComponent>("/home/magnus/phd/SuGaR/output/refined_mesh/0017/3dgs.obj");
            auto &transform = entity.getComponent<TransformComponent>();
            //transform.setScale({0.25f, 0.25f, 0.25f});
        }

        {
            auto cameraEntity = createNewCamera("DefaultCamera", 400, 400);
            auto &cameraComponent = cameraEntity.getComponent<CameraComponent>();
            cameraComponent.camera.setType(Camera::flycam);
            auto &transform = cameraEntity.getComponent<TransformComponent>();
            transform.setPosition({0.0f, 0.0f, 2.5f});
            cameraComponent.camera.pose.pos = transform.getPosition();
            cameraEntity.addComponent<MeshComponent>(1);
        }
        /*
        {
            auto cameraEntity = createNewCamera("SecondaryCamera", 1280, 720);
            auto &cameraComponent = cameraEntity.getComponent<CameraComponent>();
            cameraComponent.camera.setType(Camera::flycam);
            auto &transform = cameraEntity.getComponent<TransformComponent>();
            transform.setPosition({0.3, 0.0f, 2.5f});
            cameraComponent.camera.pose.pos = transform.getPosition();
            cameraEntity.addComponent<MeshComponent>(1);
        }
        */

        std::string colmapFilePath = "/home/magnus/Downloads/raw_data_v1_part1/0000/poses/colmap_text/images.txt";
        auto cameraPoses = loadColmapCameras(colmapFilePath);

        // Apply the camera poses to your scene
        applyColmapCameraPoses(cameraPoses);

#ifdef SYCL_ENABLED
        {auto gaussianEntity = createEntity("GaussianEntity");
                   auto &gaussianEntityModelComponent = gaussianEntity.addComponent<GaussianModelComponent>(Utils::getModelsPath() / "3dgs" / "3dgs.ply");
                   int debug = 1;
               }
#endif

    }


    void MultiSenseViewer::update(uint32_t i) {

    }


}

