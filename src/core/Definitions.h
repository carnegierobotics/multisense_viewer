//
// Created by magnus on 15/05/21.
//

#ifndef MULTISENSE_DEFINITIONS_H
#define MULTISENSE_DEFINITIONS_H

#include <utility>

#include "glm/glm.hpp"
#include "vulkan/vulkan_core.h"
#include "include/MultiSense/MultiSenseTypes.hh"

#define NUM_YUV_DATA_POINTERS 3

typedef enum CRLCameraDataType {
    AR_POINT_CLOUD,
    AR_GRAYSCALE_IMAGE,
    AR_COLOR_IMAGE_YUV420,
    AR_YUV_PLANAR_FRAME,
    AR_COLOR_IMAGE,
    CrlNone, AR_DISPARITY_IMAGE, AR_CAMERA_DATA_IMAGE
} CRLCameraDataType;

typedef enum CRLCameraType {
    DEFAULT_CAMERA_IP,
    CUSTOM_CAMERA_IP,
    VIRTUAL_CAMERA
} CRLCameraType;

typedef enum ArConnectionState {
    AR_STATE_CONNECTED,
    AR_STATE_CONNECTING,
    AR_STATE_ACTIVE,
    AR_STATE_INACTIVE,
    AR_STATE_RESET,
    AR_STATE_DISCONNECTED,
    AR_STATE_UNAVAILABLE,
    AR_STATE_JUST_ADDED
} ArConnectionState;

typedef enum StreamIndex {
    // First 0 - 8 elements also correspond to array indices. Check upon this before adding more PREVIEW indices
    AR_PREVIEW_LEFT = 0,
    AR_PREVIEW_RIGHT = 1,
    AR_PREVIEW_DISPARITY = 2,
    AR_PREVIEW_AUXILIARY = 3,
    AR_PREVIEW_POINT_CLOUD = 4,

    AR_PREVIEW_VIRTUAL_LEFT = 5,
    AR_PREVIEW_VIRTUAL_RIGHT = 6,
    AR_PREVIEW_VIRTUAL_AUX = 7,
    AR_PREVIEW_VIRTUAL_POINT_CLOUD = 8,

    AR_PREVIEW_TOTAL_MODES = AR_PREVIEW_VIRTUAL_POINT_CLOUD,
    // Other flags

} CameraStreamInfoFlag;

typedef enum CameraResolutionGroup {
    AR_GROUP_ONE,
    AR_GROUP_TWO,


} CameraResolutionGroup;

typedef enum CameraPlaybackFlags {
    AR_PREVIEW_PLAYING = 10,
    AR_PREVIEW_PAUSED = 11,
    AR_PREVIEW_STOPPED = 12,
    AR_PREVIEW_NONE = 13,
    AR_PREVIEW_RESET = 14,
} CameraPlaybackFlags;

typedef enum page {
    PAGE_PREVIEW_DEVICES = 0,
    PAGE_DEVICE_INFORMATION = 1,
    PAGE_CONFIGURE_DEVICE = 2,
    PAGE_TOTAL_PAGES = 3,
    TAB_NONE = 10,
    TAB_2D_PREVIEW = 11,
    TAB_3D_POINT_CLOUD = 12,
} Page;

typedef enum ImageElementIndex {
    IMAGE_PREVIEW_ONE = 0,
    IMAGE_INTERACTION_PREVIEW_DEVICE = 1,
    IMAGE_INTERACTION_DEVICE_INFORMATION = 2,
    IMAGE_INTERACTION_CONFIGURE_DEVICE = 3,
    IMAGE_CONFIGURE_DEVICE_PLACEHOLDER = 4,
} ImageElementIndex;

typedef enum CRLCameraResolution {
    CRL_RESOLUTION_NONE = 0,
    CRL_RESOLUTION_960_600_64 = 1,
    CRL_RESOLUTION_960_600_128 = 2,
    CRL_RESOLUTION_960_600_256 = 3,
    CRL_RESOLUTION_1920_1200_64 = 4,
    CRL_RESOLUTION_1920_1200_128 = 5,
    CRL_RESOLUTION_1920_1200_256 = 6
} CRLCameraResolution;

struct WhiteBalanceParams {
    float whiteBalanceRed = 1.0f;
    float whiteBalanceBlue = 1.0f;
    bool autoWhiteBalance = true;
    uint32_t autoWhiteBalanceDecay = 3;
    float autoWhiteBalanceThresh = 0.5f;
};

struct ExposureParams {
    bool autoExposure{};
    uint32_t exposure = 10000;
    uint32_t autoExposureMax = 5000000;
    uint32_t autoExposureDecay = 7;
    float autoExposureTargetIntensity = 0.5f;
    float autoExposureThresh = 0.9f;
    uint16_t autoExposureRoiX = 0;
    uint16_t autoExposureRoiY = 0;
    uint16_t autoExposureRoiWidth = crl::multisense::Roi_Full_Image;
    uint16_t autoExposureRoiHeight = crl::multisense::Roi_Full_Image;
    crl::multisense::DataSource exposureSource = crl::multisense::Source_Luma_Left;

};

struct EntryConnectDevice {
    std::string profileName = "MultiSense";
    std::string IP = "10.66.176.21";
    std::string interfaceName;
    uint32_t interfaceIndex{};

    std::string cameraName;

    EntryConnectDevice()= default;

    EntryConnectDevice(std::string ip, std::string iName, std::string camera, uint32_t idx) : IP(std::move(ip)),
                                                                                              interfaceName(std::move(iName)),
                                                                                              cameraName(std::move(camera)),
                                                                                              interfaceIndex(idx) {
    }

    bool ready() const{
        return ((!IP.empty() && !profileName.empty() && !interfaceName.empty() && interfaceIndex != 0) || cameraName == "Virtual Camera");
    }

};

namespace ArEngine {
    struct YUVTexture {
        void *data[NUM_YUV_DATA_POINTERS]{};
        uint32_t len[NUM_YUV_DATA_POINTERS] = {0};
        VkFlags *formats[NUM_YUV_DATA_POINTERS];
        VkFormat format{};
    };

    struct TextureData {
        void *data;
        uint32_t len;
        CRLCameraDataType type;
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
        void *plane0;
        void *plane1;
        void *plane2;

        uint32_t plane0Size;
        uint32_t plane1Size;
        uint32_t plane2Size;
    };

    struct MouseButtons {
        bool left = false;
        bool right = false;
        bool middle = false;
        float wheel = 0.0f;
    };


}


#endif //MULTISENSE_DEFINITIONS_H
