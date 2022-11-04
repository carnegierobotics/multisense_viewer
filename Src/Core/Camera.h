//
// Created by magnus on 9/4/21.
//

#ifndef MULTISENSE_CAMERA_H
#define MULTISENSE_CAMERA_H

/*
* Basic camera class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
private:
    float m_Fov = 0;
    float m_Znear = 0, m_Zfar = 1000;

    void updateViewMatrix() {
        glm::vec3 translation = m_Position;
        if (flipY) {
            translation.y *= -1.0f;
        }

        glm::vec3 dir;
        dir.x = (float) cos(glm::radians(yaw)) * (float) cos(glm::radians(pitch));
        dir.y = (float) sin(glm::radians(pitch));
        dir.z = (float) sin(glm::radians(yaw)) * (float) cos(glm::radians(pitch));
        cameraFront = glm::normalize(dir);

        if (type == CameraType::firstperson) {
            matrices.view = glm::lookAt(m_Position, m_Position + cameraFront, cameraUp);
        }
        m_ViewPos = glm::vec4(m_Position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

        updated = true;
    };
public:
    enum CameraType {
        lookat, firstperson
    };
    CameraType type = CameraType::firstperson;

    glm::vec3 m_Rotation = glm::vec3();
    glm::vec3 m_Position = glm::vec3();
    glm::vec4 m_ViewPos = glm::vec4();

    float m_RotationSpeed = 0.1f;
    float m_MovementSpeed = 1.0f;
    float m_SpeedModifier = 50.0f;

    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, -1.0f, 0.0f);
    glm::vec3 direction;
    float yaw = 0.0f, pitch = 0;


    bool updated = false;
    bool flipY = false;

    struct {
        glm::mat4 perspective{};
        glm::mat4 view{};
    } matrices{};

    struct {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    } keys;

    bool moving() {
        return keys.left || keys.right || keys.up || keys.down;
    }

    float getNearClip() {
        return m_Znear;
    }

    float getFarClip() {
        return m_Zfar;
    }

    void setPerspective(float fov, float aspect, float znear, float zfar) {
        this->m_Fov = fov;
        this->m_Znear = znear;
        this->m_Zfar = zfar;
        matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
        if (flipY) {
            matrices.perspective[1][1] *= -1.0f;
        }
    };

    void updateAspectRatio(float aspect) {
        matrices.perspective = glm::perspective(glm::radians(m_Fov), aspect, m_Znear, m_Zfar);
        if (flipY) {
            matrices.perspective[1][1] *= -1.0f;
        }
    }

    void setPosition(glm::vec3 position) {
        this->m_Position = position;
        updateViewMatrix();
    }

    void setRotation(float _yaw, float _pitch) {
        yaw = _yaw;
        pitch = _pitch;
        updateViewMatrix();
    }

    void rotate(float dx, float dy) {
        dx *= m_RotationSpeed;
        dy *= m_RotationSpeed;
        yaw -= dx;
        pitch -= dy;
        if(pitch > 89.0f)
            pitch = 89.0f;
        if(pitch < -89.0f)
            pitch = -89.0f;
        updateViewMatrix();
    }

    void setTranslation(glm::vec3 translation) {
        this->m_Position = translation;
        updateViewMatrix();
    };

    void translate(glm::vec3 delta) {
        this->m_Position += delta;
        updateViewMatrix();
    }

    void setRotationSpeed(float rotationSpeed) {
        this->m_RotationSpeed = rotationSpeed * m_SpeedModifier;
    }

    void setMovementSpeed(float movementSpeed) {
        this->m_MovementSpeed = movementSpeed * m_SpeedModifier;
    }

    void update(float deltaTime) {
        updated = false;
        if (type == CameraType::firstperson || type == CameraType::lookat) {
            if (moving()) {
                float moveSpeed = deltaTime * m_MovementSpeed;
                if (keys.up)
                    m_Position += cameraFront * moveSpeed;
                if (keys.down)
                    m_Position -= cameraFront * moveSpeed;
                if (keys.left)
                    m_Position -= glm::normalize(glm::cross(cameraFront, cameraUp)) * moveSpeed;
                if (keys.right)
                    m_Position += glm::normalize(glm::cross(cameraFront, cameraUp)) * moveSpeed;

                updateViewMatrix();
            }
        }
    };

};


#endif // MULTISENSE_CAMERA_H
