//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/VkRender/Renderer.h"

#include "Viewer/Scenes/MultiSenseViewer/MultiSenseViewer.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"

namespace VkRender {

    ColmapCamera MultiSenseViewer::loadColmapCamera(const std::string &filePath, int targetCameraId) {
        std::ifstream file(filePath);
        std::string line;
        ColmapCamera camera;

        while (std::getline(file, line)) {
            // Skip comment lines
            if (line[0] == '#') continue;

            std::istringstream iss(line);
            int cameraId;
            std::string modelName;
            int width, height;
            double focalLength, cx, cy, k1 = 0, k2 = 0;

            iss >> cameraId >> modelName >> width >> height >> focalLength >> cx >> cy;

            // Some camera models might have additional parameters (e.g., radial distortion)
            if (modelName == "SIMPLE_RADIAL" || modelName == "RADIAL") {
                iss >> k1;
            } else if (modelName == "OPENCV") {
                iss >> k1 >> k2;
            }

            if (cameraId == targetCameraId) {
                camera.cameraId = cameraId;
                camera.modelName = modelName;
                camera.width = width;
                camera.height = height;
                camera.focalLength = focalLength;
                camera.cx = cx;
                camera.cy = cy;
                camera.k1 = k1;
                camera.k2 = k2;
                break;  // We found our target camera, no need to continue
            }
        }

        return camera;
    }

    double MultiSenseViewer::computeFOV(double focalLength, double sensorSize) {
        return 2.0 * atan(sensorSize / (2.0 * focalLength)) * 180.0 / M_PI;
    }


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
    void MultiSenseViewer::applyColmapCameraPoses(const std::vector<ColmapCameraPose> &cameraPoses, double d) {
        for (const auto &pose : cameraPoses) {
            // Create or find your camera entity
            auto cameraEntity = createNewCamera(pose.imageName, 1160, 522);
            auto &cameraComponent = cameraEntity.getComponent<CameraComponent>();
            auto& fov = cameraComponent().fov();
            fov = d;
            // Set the camera's transform based on the COLMAP pose
            auto &transform = cameraEntity.getComponent<TransformComponent>();

            glm::quat colmapQuaternion(pose.rotation);
            glm::vec3 colmapTranslation(pose.translation);

            // Coordinates to the project /camera center as per colmap's documentation is given by: -R^t * T
            glm::mat3 rotMatrix = glm::mat3_cast(colmapQuaternion);

            // Extract camera coordinate system vectors
            glm::vec3 colmapRight = glm::vec3(rotMatrix[0][0], rotMatrix[1][0], rotMatrix[2][0]);
            glm::vec3 colmapUp = -glm::vec3(rotMatrix[0][1], rotMatrix[1][1], rotMatrix[2][1]); // Negating for COLMAP's Y-axis downward
            glm::vec3 colmapFront = glm::vec3(rotMatrix[0][2], rotMatrix[1][2], rotMatrix[2][2]);
            // Adjust for Vulkan view space
            glm::vec3 vulkanRight = glm::normalize(colmapRight);     // Same as COLMAP
            glm::vec3 vulkanUp = glm::normalize(-colmapUp);         // Flip Y-axis to point up
            glm::vec3 vulkanFront = glm::normalize(-colmapFront);   // Negate Z-axis to match Vulkan's forward direction


            glm::mat3 r = glm::mat3_cast(colmapQuaternion);
            glm::vec3 t = glm::vec3(pose.translation);
            glm::vec3 correctPos = glm::transpose(r) * t;
            // Construct the view matrix
            glm::mat4 viewMatrix = glm::mat4(1.0f);
            viewMatrix[0][0] = vulkanRight.x;
            viewMatrix[1][0] = vulkanRight.y;
            viewMatrix[2][0] = vulkanRight.z;
            viewMatrix[0][1] = vulkanUp.x;
            viewMatrix[1][1] = vulkanUp.y;
            viewMatrix[2][1] = vulkanUp.z;
            viewMatrix[0][2] = vulkanFront.x;
            viewMatrix[1][2] = vulkanFront.y;
            viewMatrix[2][2] = vulkanFront.z;
            viewMatrix = glm::translate(viewMatrix, correctPos);
            glm::mat4 worldMatrix = glm::inverse(viewMatrix);
            // Extract the position from the last column of the world matrix
            glm::vec3 position = glm::vec3(worldMatrix[3]);
            // Extract the rotation by taking the upper-left 3x3 part of the world matrix
            glm::mat3 rotationMatrix = glm::mat3(worldMatrix);
            // Convert the 3x3 rotation matrix to a quaternion
            glm::quat rotation = glm::normalize(glm::quat_cast(rotationMatrix));

            transform.setQuaternion(rotation);
            transform.setPosition(position);

            // Update camera component pose
            cameraComponent.camera.pose.pos = transform.getPosition();
            cameraComponent.camera.pose.q = transform.getQuaternion();
            cameraComponent.camera.matrices.view = viewMatrix;
            cameraComponent().setType(Camera::custom);

            // Optionally add other components or handle specific camera configurations here
            cameraEntity.addComponent<MeshComponent>(1);
        }
    }


    MultiSenseViewer::MultiSenseViewer(Renderer &ctx) {
        m_sceneName = "MultiSense Viewer";

        {
            auto entity = createEntity("S30Camera");
            auto &modelComponent = entity.addComponent<MeshComponent>(Utils::getModelsPath() / "obj" / "3dgs.obj");
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

        int targetCameraId = 1;  // Replace with your target camera ID

        std::string filePath = "/home/magnus/Downloads/raw_data_v1_part1/0000/poses/colmap_text/cameras.txt";

        ColmapCamera camera = loadColmapCamera(filePath, targetCameraId);

        double sensorWidth = camera.width;
        double sensorHeight = camera.height;

        double fovX = computeFOV(camera.focalLength, sensorWidth);
        double fovY = computeFOV(camera.focalLength, sensorHeight);
        double fov = std::min(fovX, fovY);
        std::string colmapFilePath = "/home/magnus/Downloads/raw_data_v1_part1/0000/poses/colmap_text/images.txt";
        auto cameraPoses = loadColmapCameras(colmapFilePath);

        // Apply the camera poses to your scene
        applyColmapCameraPoses(cameraPoses, fov);

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

