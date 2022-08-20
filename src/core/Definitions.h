//
// Created by magnus on 15/05/21.
//

#ifndef MULTISENSE_DEFINITIONS_H
#define MULTISENSE_DEFINITIONS_H

#include "glm/glm.hpp"

namespace ArEngine{
    struct VideoTexture {
        std::vector<unsigned char*> pixels{};
        uint64_t imageSize{};
        uint32_t width{};
        uint32_t height{};
    };


    struct Vertex {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv0;
        glm::vec2 uv1;
        glm::vec4 joint0;
        glm::vec4 weight0;
    };

    struct Primitive {
        uint32_t firstIndex{};
        uint32_t indexCount{};
        uint32_t vertexCount{};
        bool hasIndices{};
        //Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount);
        //void setBoundingBox(glm::vec3 min, glm::vec3 max);
    };

    struct MP4Frame {
        void* plane0;
        void* plane1;
        void* plane2;

        uint32_t plane0Size;
        uint32_t plane1Size;
        uint32_t plane2Size;
    };
}

typedef enum CRLCameraDataType {
    AR_POINT_CLOUD,
    AR_GRAYSCALE_IMAGE,
    AR_COLOR_IMAGE_YUV420,
    AR_YUV_PLANAR_FRAME,
    AR_COLOR_IMAGE,
    CrlNone, AR_DISPARITY_IMAGE, AR_CAMERA_DATA_COLOR_IMAGE
} CRLCameraDataType;

typedef enum CRLCameraType {
    DEFAULT_CAMERA_IP,
    CUSTOM_CAMERA_IP,
    VIRTUAL_CAMERA
} CRLCameraType;

typedef enum ArConnectionState{
    AR_STATE_CONNECTED,
    AR_STATE_CONNECTING,
    AR_STATE_ACTIVE,
    AR_STATE_INACTIVE,
    AR_STATE_DISCONNECTED,
    AR_STATE_UNAVAILABLE,
    AR_STATE_JUST_ADDED
} ArConnectionState;

typedef enum StreamIndex {
    // First 0 - 6 elements also correspond to array indices. Check upon this before adding more PREVIEW indices
    AR_PREVIEW_LEFT = 0,
    AR_PREVIEW_RIGHT = 1,
    AR_PREVIEW_DISPARITY = 2,
    AR_PREVIEW_AUXILIARY = 3,
    AR_PREVIEW_POINT_CLOUD = 4,
    AR_PREVIEW_VIRTUAL = 5,
    AR_PREVIEW_POINT_CLOUD_VIRTUAL = 6,
    AR_PREVIEW_TOTAL_MODES = AR_PREVIEW_POINT_CLOUD_VIRTUAL,
    // Other flags
    AR_PREVIEW_PLAYING = 10,
    AR_PREVIEW_PAUSED = 11,
    AR_PREVIEW_STOPPED = 12,
    AR_PREVIEW_NONE = 13,
    AR_PREVIEW_RESET = 14,
} CameraStreamInfoFlag;

typedef enum page {
    PAGE_PREVIEW_DEVICES = 0,
    PAGE_DEVICE_INFORMATION = 1,
    PAGE_CONFIGURE_DEVICE = 2,
    PAGE_TOTAL_PAGES = 3,
    TAB_NONE = 10,
    TAB_2D_PREVIEW = 11,
    TAB_3D_POINT_CLOUD = 12,
} Page;


#endif //MULTISENSE_DEFINITIONS_H
