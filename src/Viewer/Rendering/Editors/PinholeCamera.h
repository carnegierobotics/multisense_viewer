//
// Created by magnus-desktop on 12/11/24.
//

#ifndef PINHOLECAMERA_H
#define PINHOLECAMERA_H

#include <multisense_viewer/src/Viewer/Rendering/Editors/BaseCamera.h>


namespace VkRender {
    class PinholeCamera : public BaseCamera {
    public:
        float m_fx, m_fy, m_cx, m_cy;
        float m_width, m_height;
        PinholeCamera(uint32_t width, uint32_t height, float fx, float fy, float cx, float cy, float zNear=0.1f, float zFar=10.0f) {
            m_width = static_cast<float>(width);
            m_height =static_cast<float>(height);
            m_fx = fx; m_fy = fy; m_cx = cx; m_cy = cy;
            m_zNear = zNear;
            m_zFar = zFar;
            PinholeCamera::updateProjectionMatrix();
        }

        void updateProjectionMatrix() override {
            float A = -m_zFar / (m_zFar - m_zNear);
            float B = (-m_zFar * m_zNear) / (m_zFar - m_zNear);
            float w = m_width;
            float h = m_height;

            matrices.projection = glm::mat4(
                (2.0f * m_fx) / w,       0.0f,                     ((2.0f * m_cx) - w) / w,      0.0f,
                0.0f,                   (2.0f * m_fy) / h,         ((2.0f * m_cy) - h) / h,      0.0f,
                0.0f,                    0.0f,                     A,                           -1.0f,
                0.0f,                    0.0f,                     B,                            0.0f
            );
        }
    };
}
#endif //PINHOLECAMERA_H