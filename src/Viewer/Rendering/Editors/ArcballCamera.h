//
// Created by magnus-desktop on 12/11/24.
//

#ifndef ARCBALLCAMERA_H
#define ARCBALLCAMERA_H
namespace VkRender {
    class ArcballCamera : public BaseCamera {
    public:
        glm::vec3 target = glm::vec3(0.0f);
        float distance = 5.0f;
        float rotationSpeed = 0.1f;
        float zoomSpeed = 0.1f;

        ArcballCamera(float aspect, TransformComponent& transform)
            : BaseCamera(aspect), m_transform(transform) {
            updateViewMatrix();
        }

        void handleInput(const InputState& input) {
            if (input.isRotating) {
                float yaw = input.deltaX * rotationSpeed;
                float pitch = input.deltaY * rotationSpeed;

                glm::quat qYaw = glm::angleAxis(glm::radians(yaw), glm::vec3(0.0f, 0.0f, 1.0f));
                glm::quat qPitch = glm::angleAxis(glm::radians(pitch), glm::vec3(1.0f, 0.0f, 0.0f));
                pose.orientation = glm::normalize(qYaw * pose.orientation * qPitch);
                pose.updateVectors();
            }

            if (input.scrollDelta != 0.0f) {
                distance -= input.scrollDelta * zoomSpeed;
                if (distance < 1.0f) distance = 1.0f;
            }

            // Update the camera's position based on target and distance
            glm::vec3 direction = glm::normalize(pose.position - target);
            pose.position = target + direction * distance;

            // Update the TransformComponent
            m_transform.setPosition(pose.position);
            m_transform.setRotationQuaternion(pose.orientation);

            // Update the view matrix
            updateViewMatrix();
        }

    private:
        TransformComponent& m_transform;
    };
}
#endif //ARCBALLCAMERA_H
