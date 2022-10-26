//
// Created by magnus on 10/25/22.
//

#ifndef MULTISENSE_VIEWER_TEXTUREDATADEF_H
#define MULTISENSE_VIEWER_TEXTUREDATADEF_H

#include <cstdint>
#include <cstdlib>
#include "MultiSense/Src/Core/Definitions.h"
#include "MultiSense/Src/Tools/Utils.h"

namespace VkRender {
    /**
    * @brief Container for bridging the gap between a VkRender Frame and the viewer render resources
    */
    struct TextureData {
        /**
         * @brief Default Constructor. Sizes between texture and camera frame must match!
         * Can use a memory pointer to GPU memory as memory store option or handle memory manually with \refitem manualMemoryMgmt.
         * @param texType Which texture type this is to be used with. Used to calculate the size of the texture
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
                data = (uint8_t *) malloc(m_Len);
                data2 = (uint8_t *) malloc(m_Len2);
            }
        }

        TextureData(CRLCameraDataType type, CRLCameraResolution resolution, bool manualMemoryMgmt = false) :
                m_Type(type),
                m_Res(resolution),
                m_Manual(manualMemoryMgmt) {

            uint32_t depth = 0;
            Utils::cameraResolutionToValue(m_Res, &m_Width, &m_Height, &depth);
            calcImageSize();
        }

        ~TextureData() {
            if (m_Manual) {
                free(data);
                free(data2);
            }
        }

        CRLCameraDataType m_Type = AR_CAMERA_IMAGE_NONE;
        CRLCameraResolution m_Res = CRL_RESOLUTION_NONE;
        uint32_t m_Width = 0;
        uint32_t m_Height = 0;
        uint8_t *data{};
        uint8_t *data2{};
        uint32_t m_Len = 0, m_Len2 = 0, m_Id{}, m_Id2{};

    private:
        bool m_Manual = true;
        void calcImageSize(){
            switch (m_Type) {
                case AR_POINT_CLOUD:
                case AR_GRAYSCALE_IMAGE:
                    m_Len = m_Width * m_Height;
                    break;
                case AR_COLOR_IMAGE_YUV420:
                case AR_YUV_PLANAR_FRAME:
                    m_Len = m_Width * m_Height;
                    m_Len2 = m_Width * m_Height / 2;
                    break;
                case AR_DISPARITY_IMAGE:
                    m_Len = m_Width * m_Height * 2;
                    break;
                default:
                    break;
            }
        }

    };
}
#endif //MULTISENSE_VIEWER_TEXTUREDATADEF_H
