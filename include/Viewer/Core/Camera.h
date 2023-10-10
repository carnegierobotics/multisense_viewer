/**
 * @file: MultiSense-Viewer/include/Viewer/Core/Camera.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2021-9-4, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_CAMERA_H
#define MULTISENSE_CAMERA_H

#include <glm/gtx/string_cast.hpp>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define PI 3.14159265359f


#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace VkRender {
    class Camera {
    private:
        float m_Fov = 0;
        float m_Znear = 0, m_Zfar = 100;

        void updateViewMatrix() {
            if (type == CameraType::flycam) {
                matrices.view = glm::lookAt(m_Position, m_Position + cameraFront, cameraUp);
// Convert to direction vector (assuming initial direction is along X-axis)

                // matrices.view = glm::rotate(matrices.view, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                // m_ViewPos = glm::vec4(m_Position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);
            } else if (type == CameraType::arcball) {



                // 1. Translate the scene so that the camera's position becomes the origin
                // 2. Apply rotations
                glm::mat4 rotM = glm::mat4(1.0f);
                rotM = glm::rotate(rotM, glm::radians(m_Rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));  // X-axis rotation
                rotM = glm::rotate(rotM, glm::radians(m_Rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));  // Z-axis rotation

                // 4. Translate the camera based on the zoom value

                glm::mat4 transM = glm::translate(glm::mat4(1.0f), zoomVal * m_Position);
                glm::mat4 transMat = glm::translate(glm::mat4(1.0f), m_Translate);

                matrices.view = transM * rotM * transMat;

            }

        };
    public:
        enum CameraType {
            arcball, flycam
        };
        CameraType type = CameraType::flycam;

        glm::vec3 m_Rotation = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 m_Position = glm::vec3(0.0f, 0.0f, 0.0f);

        glm::vec3 m_Translate = glm::vec3(0.0f, 0.0f, 0.0f);

        glm::vec3 cameraFront = glm::vec3(0.0f, -1.0f, 0.0f);
        glm::vec3 cameraUp = glm::vec3(0.0f, 0.0f, 1.0f);
        glm::vec3 cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);

        glm::vec3 arcBallTranslate = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 arcballFront = glm::vec3(1.0f, 0.0f, 0.0f);

        float m_RotationSpeed = 0.20f;
        float m_MovementSpeed = 3.0f;
        glm::quat orientation = glm::angleAxis(glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        float zoomVal = 1.0f;
        struct {
            glm::mat4 perspective = glm::mat4(1.0f);
            glm::mat4 view = glm::mat4(1.0f);
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
            matrices.perspective[1][1] *= -1;
        };

        void updateAspectRatio(float aspect) {
            matrices.perspective = glm::perspective(glm::radians(m_Fov), aspect, m_Znear, m_Zfar);
            matrices.perspective[1][1] *= -1;
        }

        void resetPosition() {
            glm::vec3 pos(-3.0f, 0.0f, 1.50f);
            m_Translate = glm::vec3(0.0f, 0.0f, 0.0f);

            if (type == arcball) {
                this->m_Position = pos * glm::vec3(0.0f, 0.0f, -1.0f); // Setting for arcball we just want a Z value
            } else
                this->m_Position = pos;

            updateViewMatrix();
        }


        void resetRotation() {
            m_Rotation = glm::vec3(-65.0f, 0.0f, 90.0f);
            orientation = glm::angleAxis(glm::radians(181.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            rotate(0, 0);
            updateViewMatrix();
        }

        void setArcBallPosition(float f) {
            if (type == arcball) {

                zoomVal *= std::abs((f));
                updateViewMatrix();
            }
        }

        void rotate(float dx, float dy) {
            dx *= m_RotationSpeed;
            dy *= m_RotationSpeed;

            if (type == flycam) {
                // On mouse move:
                // Calculate yaw rotation
                glm::quat yawRotation = glm::angleAxis(glm::radians(dx / 2.0f),
                                                       cameraUp); // Rotate around Z-axis

                // Apply yaw rotation to the current orientation
                orientation = yawRotation * orientation;

                // Extract the camera's local right (X) axis after yaw rotation
                // glm::vec3 localRight = glm::mat3_cast(orientation) * glm::vec3(1.0f, 0.0f, 0.0f);
                cameraRight = glm::cross(cameraUp, cameraFront);
                glm::quat pitchRotation = glm::angleAxis(glm::radians(-dy / 2.0f), cameraRight);

                // Combine the rotations
                orientation = pitchRotation * orientation;
                orientation = glm::normalize(orientation);  // Ensure it stays normalized

                Log::Logger::getInstance()->info("Orientation {},{},{},{}", orientation.x, orientation.y, orientation.z,
                                                 orientation.w);

                // Extract the camera's front direction
                glm::vec3 dir = glm::mat3_cast(orientation) * glm::vec3(1.0f, 0.0f, 0.0f);
                cameraFront = dir;
            } else {
                m_Rotation.x += dy;
                m_Rotation.z += dx;
            }
            updateViewMatrix();
            translate(glm::vec3(0.0f));
        }

        void translate(glm::vec3 delta) {
            delta.y *= -1;
            glm::mat4 rotM = glm::mat4(1.0f);
            rotM = glm::rotate(rotM, glm::radians(m_Rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));  // X-axis rotation
            rotM = glm::rotate(rotM, glm::radians(m_Rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));  // Z-axis rotation
            glm::vec4 rot = glm::vec4(delta, 1.0f) * rotM;
            m_Translate -= glm::vec3(rot);

            updateViewMatrix();
        }

        void translate(float dx, float dy) {
            glm::vec3 right = glm::normalize(glm::cross(cameraFront, glm::vec3(0.0f, 0.0f, 1.0f)));  // Assuming Z is up
            glm::vec3 up = glm::normalize(glm::cross(right, cameraFront));

            m_Position += right * dx;  // Pan right/left based on mouse x-delta
            m_Position += up * (-dy);     // Pan up/down based on mouse y-delta

            updateViewMatrix();
        }

        void update(float deltaTime) {
            if (type == CameraType::flycam) {
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
}

#endif // MULTISENSE_CAMERA_H
