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

class Camera
{
private:
    float m_Fov = 0;
    float m_Znear = 0, m_Zfar = 1000;

    void updateViewMatrix()
    {
        glm::mat4 rotM = glm::mat4(1.0f);
        glm::mat4 transM;

        rotM = glm::rotate(rotM, glm::radians(m_Rotation.x * (flipY ? -1.0f : 1.0f)), glm::vec3(1.0f, 0.0f, 0.0f));
        rotM = glm::rotate(rotM, glm::radians(m_Rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        rotM = glm::rotate(rotM, glm::radians(m_Rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

        glm::vec3 translation = m_Position;
        if (flipY) {
            translation.y *= -1.0f;
        }
        transM = glm::translate(glm::mat4(1.0f), translation);

        if (type == CameraType::firstperson)
        {
            matrices.view = rotM * transM;
            matrices.view = glm::lookAt(m_Position, m_Position + cameraFront, cameraUp);

        }
        else
        {
            matrices.view = transM * rotM;
        }

        m_ViewPos = glm::vec4(m_Position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

        updated = true;
    };
public:
    enum CameraType { lookat, firstperson };
    CameraType type = CameraType::firstperson;

    glm::vec3 m_Rotation = glm::vec3();
    glm::vec3 m_Position = glm::vec3();
    glm::vec4 m_ViewPos = glm::vec4();

    float m_RotationSpeed = 1.0f;
    float m_MovementSpeed = 1.0f;
    float m_SpeedModifier = 50.0f;

    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp    = glm::vec3(0.0f, -1.0f,  0.0f);


    bool updated = false;
    bool flipY = false;

    struct
    {
        glm::mat4 perspective{};
        glm::mat4 view{};
    } matrices{};

    struct
    {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    } keys;

    bool moving()
    {
        return keys.left || keys.right || keys.up || keys.down;
    }

    float getNearClip() {
        return m_Znear;
    }

    float getFarClip() {
        return m_Zfar;
    }

    void setPerspective(float fov, float aspect, float znear, float zfar)
    {
        this->m_Fov = fov;
        this->m_Znear = znear;
        this->m_Zfar = zfar;
        matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
        if (flipY) {
            matrices.perspective[1][1] *= -1.0f;
        }
    };

    void updateAspectRatio(float aspect)
    {
        matrices.perspective = glm::perspective(glm::radians(m_Fov), aspect, m_Znear, m_Zfar);
        if (flipY) {
            matrices.perspective[1][1] *= -1.0f;
        }
    }

    void setPosition(glm::vec3 position)
    {
        this->m_Position = position;
        updateViewMatrix();
    }

    void setRotation(glm::vec3 rotation)
    {
        this->m_Rotation = rotation;
        updateViewMatrix();
    }

    void rotate(glm::vec3 delta)
    {
        this->m_Rotation += delta;
        updateViewMatrix();
    }

    void setTranslation(glm::vec3 translation)
    {
        this->m_Position = translation;
        updateViewMatrix();
    };

    void translate(glm::vec3 delta)
    {
        this->m_Position += delta;
        updateViewMatrix();
    }

    void setRotationSpeed(float rotationSpeed)
    {
        this->m_RotationSpeed = rotationSpeed * m_SpeedModifier;
    }

    void setMovementSpeed(float movementSpeed)
    {
        this->m_MovementSpeed = movementSpeed * m_SpeedModifier;
    }

    void update(float deltaTime)
    {
        updated = false;
        if (type == CameraType::firstperson || type == CameraType::lookat)
        {
            if (moving())
            {
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

    // Update camera passing separate axis data (gamepad)
    // Returns true if m_View or m_Position has been changed
    bool updatePad(glm::vec2 axisLeft, glm::vec2 axisRight, float deltaTime)
    {
        bool retVal = false;

        if (type == CameraType::firstperson)
        {
            // Use the common console thumbstick layout
            // Left = m_View, right = move

            const float deadZone = 0.0015f;
            const float range = 1.0f - deadZone;

            glm::vec3 camFront{};
            camFront.x = -cos(glm::radians(m_Rotation.x)) * sin(glm::radians(m_Rotation.y));
            camFront.y = sin(glm::radians(m_Rotation.x));
            camFront.z = cos(glm::radians(m_Rotation.x)) * cos(glm::radians(m_Rotation.y));
            camFront = glm::normalize(camFront);

            float moveSpeed = deltaTime * m_MovementSpeed * 2.0f;
            float rotSpeed = deltaTime * m_RotationSpeed * 50.0f;

            // Move
            if (fabsf(axisLeft.y) > deadZone)
            {
                float pos = (fabsf(axisLeft.y) - deadZone) / range;
                m_Position -= camFront * pos * ((axisLeft.y < 0.0f) ? -1.0f : 1.0f) * moveSpeed;
                retVal = true;
            }
            if (fabsf(axisLeft.x) > deadZone)
            {
                float pos = (fabsf(axisLeft.x) - deadZone) / range;
                m_Position += glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) * pos * ((axisLeft.x < 0.0f) ? -1.0f : 1.0f) * moveSpeed;
                retVal = true;
            }

            // Rotate
            if (fabsf(axisRight.x) > deadZone)
            {
                float pos = (fabsf(axisRight.x) - deadZone) / range;
                m_Rotation.y += pos * ((axisRight.x < 0.0f) ? -1.0f : 1.0f) * rotSpeed;
                retVal = true;
            }
            if (fabsf(axisRight.y) > deadZone)
            {
                float pos = (fabsf(axisRight.y) - deadZone) / range;
                m_Rotation.x -= pos * ((axisRight.y < 0.0f) ? -1.0f : 1.0f) * rotSpeed;
                retVal = true;
            }
        }
        else
        {

        }

        if (retVal)
        {
            updateViewMatrix();
        }

        return retVal;
    }

};



#endif // MULTISENSE_CAMERA_H
