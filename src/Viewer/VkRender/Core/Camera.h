/**
 * @file: MultiSense-Viewer/include/Viewer/VkRender/Core/Camera.h
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

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define PI 3.14159265359f


#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/gtx/string_cast.hpp>

#include "Viewer/Tools/Logger.h"

#define DEFAULT_FRONT glm::vec3(0.0f, 0.0f, -1.0f)
#define DEFAULT_UP glm::vec3(0.0f, 1.0f, 0.0f)
#define DEFAULT_RIGHT glm::vec3(1.0f, 0.0f, 0.0f)
//#define DEFAULT_ORIENTATION glm::quat(0.780483f, 0.483536f, 0.208704f, 0.336872f)
#define DEFAULT_ORIENTATION glm::quat(1.0f, 0.0f, 0.0f, 0.0f)
//#define DEFAULT_POSITION glm::vec3(7.35f, -6.9f, 4.9f)
#define DEFAULT_POSITION glm::vec3(0.0f, 0.0f, 3.0f);

namespace VkRender {
    class Camera {
    public:
        enum CameraType {
            arcball, flycam
        };
        CameraType m_type = CameraType::arcball;

        float m_RotationSpeed = 0.20f;
        float m_MovementSpeed = 5.0f;
        float zoomVal = 1.0f;
        glm::vec2 rot = glm::vec2(0.0f, 0.0f);
        struct {
            glm::mat4 perspective = glm::mat4(1.0f);
            glm::mat4 view = glm::mat4(1.0f);
        } matrices{};


        float m_Fov = 60.0f;
        float m_Znear = 0.1f;
        float m_Zfar = 10.0f;

        uint32_t m_height = 0;
        uint32_t m_width = 0;

        Camera() = default;

        Camera(uint32_t width, uint32_t height) {
            m_width = width;
            m_height = height;
            m_type = VkRender::Camera::arcball;
            setPerspective(60.0f, static_cast<float>(width) / static_cast<float>(height));
            resetPosition();
            // Initialize quaternion to have a forward looking x-axis
            //rotateQuaternion(-90.0f, glm::vec3(0.0f, 1.0f, 0.0f));
            if (m_height == 0 || m_width == 0){
                Log::Logger::getInstance()->warning("Initializing a camera with 0 on one dimension {}x{}", m_width, m_height);
            }
        }

        void updateViewMatrix() {
            if (m_type == CameraType::flycam) {
                matrices.view = glm::inverse(getFlyCameraTransMat());
            } else if (m_type == CameraType::arcball) {
                matrices.view = glm::inverse(getArcBallCameraTransMat());
            }

        };

        void setType(CameraType type) {
            m_type = type;
        }


        struct Pose {
            //glm::quat q = glm::quat(0.5f, 0.5f, -0.5f, -0.5f); // We start by having a orientation facing positive x
            glm::quat q = DEFAULT_ORIENTATION; // We start by having a orientation facing positive x
            glm::vec3 pos = DEFAULT_POSITION; // default starting location
            glm::vec3 front = DEFAULT_FRONT; // Default Vulkan is negative-z is forward
            glm::vec3 up = DEFAULT_UP;
            glm::vec3 right = DEFAULT_RIGHT;

            void updateVectors() {
                // Rotate the base vectors according to the current orientation
                front = glm::normalize(glm::mat3_cast(q) * DEFAULT_FRONT);
                up = glm::normalize(glm::mat3_cast(q) * DEFAULT_UP);
                right = glm::normalize(glm::mat3_cast(q) * DEFAULT_RIGHT);
            }

            void reset() {
                q = DEFAULT_ORIENTATION;
                pos = DEFAULT_POSITION;
                front = DEFAULT_FRONT;
                up = DEFAULT_UP;
                right = DEFAULT_RIGHT;
            }
        } pose;

        glm::mat4 getFlyCameraTransMat() {
            // This function constructs a 4x4 matrix from a quaternion.
            glm::mat4 rotMatrix = glm::mat4_cast(pose.q);
            glm::mat4 transMatrix = glm::translate(glm::mat4(1.0f), pose.pos);
            //Log::Logger::getInstance()->info("Camera: {}, {}, {}", pose.pos.x, pose.pos.y, pose.pos.z);
            glm::mat4 transformationMatrix = transMatrix * rotMatrix;
            return transformationMatrix;
        }

        glm::mat4 getArcBallCameraTransMat() {
            glm::mat4 transMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 3.0f) * zoomVal);
            // Convert quaternion to rotation matrix
            glm::mat4 rotMatrix = glm::mat4_cast(pose.q);
            auto trans = rotMatrix * transMatrix;
            pose.pos = glm::vec3(trans[3]); // update actual position of camera
            pose.q = glm::quat_cast(trans);
            return trans;

        }

        /**
         * @brief normalizes the axis before performing rotation to orientation quat
         * @param angle
         * @param axis
         */
        void rotateQuaternion(float angle, glm::vec3 axis) {
            auto R = glm::angleAxis(glm::radians(angle), glm::normalize(axis));
            pose.q = glm::normalize(R * pose.q);
        }

        void rotateArcBall(float dx, float dy) {
            pose.q = DEFAULT_ORIENTATION;
            // Adjust rotation based on the mouse movement
            glm::quat rotX = glm::angleAxis(glm::radians(dx), glm::vec3(0.0f, 0.0f, 1.0f));
            glm::quat rotY = glm::angleAxis(glm::radians(dy), glm::vec3(1.0f, 0.0f, 0.0f));

            // Combine rotations in a specific order
            pose.q = rotX * pose.q;
            pose.q = pose.q * rotY;

            // Normalize the pose.q quaternion to avoid floating-point drift
            pose.q = glm::normalize(pose.q);

            pose.updateVectors();
        }

        void rotate(float dx, float dy) {
            dx *= m_RotationSpeed;
            dy *= m_RotationSpeed;

            if (m_type == arcball) {
                rot.x += dx;
                rot.y += dy;
                rotateArcBall(rot.x, rot.y);
            } else {
                rotateQuaternion(dy, pose.right);
                rotateQuaternion(dx, glm::vec3(0.0f, 0.0f, 1.0f));
                pose.updateVectors();
            }
            updateViewMatrix();
        }

        struct {
            bool left = false;
            bool right = false;
            bool up = false;
            bool down = false;
        } keys;


        bool moving() {
            return keys.left || keys.right || keys.up || keys.down;
        }


        void setPerspective(float fov, float aspect, float zNear = 0.1f, float zFar = 40.0f) {
            // Guide: https://vincent-p.github.io/posts/vulkan_perspective_matrix/

            m_Fov = fov;
            m_Znear = zNear;
            m_Zfar = zFar;
            float focalLength = 1.0f / tanf(glm::radians(m_Fov) * 0.5f);
            float x = focalLength / aspect;
            float y = -focalLength;
            float A = -m_Zfar / (m_Zfar - m_Znear);
            float B = (-m_Zfar * m_Znear) / (m_Zfar - m_Znear);
            matrices.perspective = glm::mat4(
                    x, 0.0f, 0.0f, 0.0f,
                    0.0f, y, 0.0f, 0.0f,
                    0.0f, 0.0f, A, -1.0f,
                    0.0f, 0.0f, B, 0.0f
            );
            /*
            float right = 1;
            float left = -1;
            float rightMinusLeft = right - left;
            float top = -1;
            float bottom = 1;
            float bottomMinusTop = bottom - top;
            float twoNear = 2 * m_Znear;
            glm::mat4 m = {
                    twoNear / rightMinusLeft, 0.0f, 0.0f, 0.0f,
                    0.0f, -twoNear / bottomMinusTop, 0.0f, 0.0f,
                    0.0f, 0.0f, A, -1.0f,
                    0.0f, 0.0f, B, 0.0f
            };
            matrices.perspective = m;
             */
        };

        void updateAspectRatio(float aspect) {
            float focal_length = 1.0f / tanf(glm::radians(m_Fov) * 0.5f);
            float x = focal_length / aspect;
            float y = -focal_length;
            float A = -m_Zfar / (m_Zfar - m_Znear);
            float B = -m_Zfar * m_Znear / (m_Zfar - m_Znear);

            matrices.perspective = glm::mat4(
                    x, 0.0f, 0.0f, 0.0f,
                    0.0f, y, 0.0f, 0.0f,
                    0.0f, 0.0f, A, -1.0f,
                    0.0f, 0.0f, B, 0.0f
            );
        }

        void resetPosition() {
            pose.reset();
            pose.updateVectors();
            rot = glm::vec2(0.0f);
            updateViewMatrix();
        }

        void setArcBallPosition(float f) {
            if (m_type == arcball) {
                zoomVal *= std::abs((f));
                updateViewMatrix();
            }
        }

        void translate(glm::vec3 delta) {
            delta.y *= -1;
            //glm::mat4 rotM = glm::mat4(1.0f);
            //rotM = glm::rotate(rotM, glm::radians(m_Rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));  // X-axis rotation
            //rotM = glm::rotate(rotM, glm::radians(m_Rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));  // Z-axis rotation
            //glm::vec4 rot = glm::vec4(delta, 1.0f) * rotM;
            //m_Translate -= glm::vec3(rot);

            updateViewMatrix();
        }

        void translate(float dx, float dy) {
            glm::vec3 right = glm::normalize(
                    glm::cross(pose.front, glm::vec3(0.0f, 0.0f, 1.0f)));  // Assuming Z is up
            glm::vec3 up = glm::normalize(glm::cross(right, pose.front));

            pose.pos += right * dx;  // Pan right/left based on mouse x-delta
            pose.pos += up * (-dy);     // Pan up/down based on mouse y-delta

            updateViewMatrix();
        }

        void update(float deltaTime) {
            if (m_type == CameraType::flycam) {
                if (moving()) {
                    float moveSpeed = deltaTime * m_MovementSpeed;
                    if (keys.up) {
                        pose.pos += pose.front * moveSpeed;
                    }
                    if (keys.down) {
                        pose.pos -= pose.front * moveSpeed;
                    }
                    if (keys.left) {
                        pose.pos -= glm::normalize(glm::cross(pose.front, pose.up)) * moveSpeed;
                    }
                    if (keys.right) {
                        pose.pos += glm::normalize(glm::cross(pose.front, pose.up)) * moveSpeed;
                    }

                    updateViewMatrix();
                }
            }
        };
    };
}

#endif // MULTISENSE_CAMERA_H
