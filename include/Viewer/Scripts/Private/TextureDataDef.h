/**
 * @file: MultiSense-Viewer/include/Viewer/Scripts/Private/TextureDataDef.h
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
 *   2022-10-25, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_VIEWER_TEXTUREDATADEF_H
#define MULTISENSE_VIEWER_TEXTUREDATADEF_H

#include <cstdint>
#include <cstdlib>

#include "Viewer/Core/Definitions.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender {
    /**
    * @brief Container for bridging the gap between a VkRender Frame and the viewer render resources
    */
    struct TextureData {
        /**
         * @brief Default Constructor. Sizes between texture and camera frame must match!
         * Can use a memory pointer to GPU memory as memory store option or handle memory manually with \refitem manualMemoryMgmt.
         * @param texType Which texture type this is to be used with. Used to calculate the m_TexSize of the texture
         * @param width Width of texture/frame
         * @param height Width of texture/frame
         * @param manualMemoryMgmt true: Malloc memory with this object, false: dont malloc memory, default = false
         */
        explicit TextureData(CRLCameraDataType texType, uint32_t width, uint32_t height, bool manualMemoryMgmt = false)
                :
                m_Type(texType),
                m_Width(width),
                m_Height(height),
                m_Manual(manualMemoryMgmt) {

            calcImageSize();

            if (m_Manual) {
                data = reinterpret_cast<uint8_t *>(malloc(m_Len));
                data2 = reinterpret_cast<uint8_t *>(malloc(m_Len2));
            }
        }

        TextureData(CRLCameraDataType type, CRLCameraResolution resolution, bool manualMemoryMgmt = false) :
                m_Type(type),
                m_Res(resolution),
                m_Manual(manualMemoryMgmt) {

            uint32_t depth = 0;
            Utils::cameraResolutionToValue(m_Res, &m_Width, &m_Height, &depth);
            calcImageSize();

            if (m_Manual) {
                data = reinterpret_cast<uint8_t *>(malloc(m_Len));
                data2 = reinterpret_cast<uint8_t *>(malloc(m_Len2));
            }
        }

        ~TextureData() {
            if (m_Manual) {
                free(data);
                free(data2);
            }
        }

        CRLCameraDataType m_Type = CRL_CAMERA_IMAGE_NONE;
        CRLCameraResolution m_Res = CRL_RESOLUTION_NONE;
        uint32_t m_Width = 0;
        uint32_t m_Height = 0;
        uint8_t* data{};
        uint8_t* data2{};
        uint8_t* data3 = nullptr;
        uint32_t m_Len = 0, m_Len2 = 0, m_Id{}, m_Id2{};

    private:
        bool m_Manual = true;
        void calcImageSize(){
            switch (m_Type) {
                case CRL_POINT_CLOUD:
                case CRL_GRAYSCALE_IMAGE:
                    m_Len = m_Width * m_Height;
                    break;
                case CRL_COLOR_IMAGE_YUV420:
                case CRL_YUV_PLANAR_FRAME:
                    m_Len = m_Width * m_Height;
                    m_Len2 = m_Width * m_Height / 2;
                    break;
                case CRL_DISPARITY_IMAGE:
                    m_Len = m_Width * m_Height * 2;
                    break;
                default:
                    break;
            }
        }

    };

    /**
     * Radians
     */
    struct IMUData{
        double roll = 0.0f;
        double pitch = 0.0f;
    };

    struct Shared{
        size_t frame{};
        std::string time;
    };
}
#endif //MULTISENSE_VIEWER_TEXTUREDATADEF_H
