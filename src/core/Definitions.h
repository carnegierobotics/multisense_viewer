//
// Created by magnus on 15/05/21.
//

#ifndef MULTISENSE_DEFINITIONS_H
#define MULTISENSE_DEFINITIONS_H


#include "glm/glm.hpp"
#include "vulkan/vulkan_core.h"
#include "include/MultiSense/MultiSenseTypes.hh"
#include "MultiSense/src/tools/Logger.h"
#include "map"
#include <utility>

#define NUM_YUV_DATA_POINTERS 3
#define NUM_POINTS 2048 // Changing this also needs to be changed in the vs shader.
#define INTERVAL_10_SECONDS_LOG_DRAW_COUNT 10

typedef enum ScriptType {
    AR_SCRIPT_TYPE_DISABLED,
    AR_SCRIPT_TYPE_DEFAULT,
    AR_SCRIPT_TYPE_CRL_CAMERA,
    AR_SCRIPT_TYPE_CRL_CAMERA_SETUP_ONLY,
    AR_SCRIPT_TYPE_POINT_CLOUD
} ScriptType;


typedef enum CRLCameraDataType {
    AR_POINT_CLOUD,
    AR_GRAYSCALE_IMAGE,
    AR_COLOR_IMAGE_YUV420,
    AR_YUV_PLANAR_FRAME,
    AR_CAMERA_IMAGE_NONE,
    AR_DISPARITY_IMAGE,
} CRLCameraDataType;

typedef enum CRLCameraType {
    DEFAULT_CAMERA_IP,
    CUSTOM_CAMERA_IP,
    VIRTUAL_CAMERA
} CRLCameraType;

typedef enum ArConnectionState {
    AR_STATE_CONNECTED = 0,
    AR_STATE_CONNECTING = 1,
    AR_STATE_ACTIVE = 2,
    AR_STATE_INACTIVE = 3,
    AR_STATE_RESET = 4,
    AR_STATE_DISCONNECTED = 5,
    AR_STATE_UNAVAILABLE = 6,
    AR_STATE_JUST_ADDED = 7
} ArConnectionState;

typedef enum StreamIndex {
    // First 0 - 8 elements also correspond to array indices. Check upon this before adding more PREVIEW indices
    AR_PREVIEW_ONE = 0,
    AR_PREVIEW_TWO = 1,
    AR_PREVIEW_THREE = 2,
    AR_PREVIEW_FOUR = 3,
    AR_PREVIEW_FIVE = 5,
    AR_PREVIEW_SIX = 6,
    AR_PREVIEW_SEVEN = 7,
    AR_PREVIEW_EIGHT = 8,
    AR_PREVIEW_NINE = 9,
    AR_PREVIEW_POINT_CLOUD = 10,


    AR_PREVIEW_TOTAL_MODES = AR_PREVIEW_POINT_CLOUD,
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
    CRL_RESOLUTION_1920_1200_256 = 6,
    CRL_RESOLUTION_TOTAL_MODES = 7,
} CRLCameraResolution;

typedef enum PreviewLayout {
    PREVIEW_LAYOUT_NONE = 0,
    PREVIEW_LAYOUT_SINGLE = 1,
    PREVIEW_LAYOUT_DOUBLE = 2,
    PREVIEW_LAYOUT_DOUBLE_SIDE_BY_SIDE = 3,
    PREVIEW_LAYOUT_QUAD = 4,
    PREVIEW_LAYOUT_NINE = 5
} PreviewLayout;

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


namespace AR {

/** @brief  */
    struct StreamingModes {
        /** @brief Name of this streaming mode (i.e: front camera) */
        std::string name;
        std::string attachedScript;
        /** @brief Which gui index is selected */
        CameraStreamInfoFlag streamIndex = AR_PREVIEW_ONE;
        /** @brief Current camera streaming state  */
        CameraPlaybackFlags playbackStatus = AR_PREVIEW_NONE;
        /** @brief In which order is this streaming mode presented in the viewing area  */
        uint32_t streamingOrder = 0;
        /** @brief Camera streaming modes  */
        std::vector<std::string> modes;
        /** @brief Camera streaming sources  */
        std::vector<std::string> sources;
        uint32_t selectedModeIndex = 0;
        uint32_t selectedSourceIndex = 0;
        /** @brief Which mode is currently selected */
        CRLCameraResolution selectedStreamingMode{};
        /** @brief Which source is currently selected */
        std::string selectedStreamingSource = "Select sensor resolution";

    };


/** @brief  */
    struct Parameters {
        bool initialized = false;
        float gain = 1.0f;
        float fps = 30.0f;
        ExposureParams ep;
        WhiteBalanceParams wb;
        float stereoPostFilterStrength = 0.5f;
        bool hdrEnabled;
        bool storeSettingsInFlash;
        crl::multisense::CameraProfile cameraProfile;
        float gamma = 2.0f;
        bool update = false;
    };

    struct CursorPixelInformation {
        uint32_t x, y;
        uint32_t r, g, b;
        uint32_t intensity;
        uint32_t depth;
    };


    struct Element {
        /** @brief Profile Name information  */
        std::string name = "Profile #1"; // TODO Remove if remains Unused
        /** @brief Identifier of the camera connected  */
        std::string cameraName;
        /** @brief IP of the connected camera  */
        std::string IP;
        /** @brief Default IP of the connected camera  */
        std::string defaultIP = "10.66.171.21"; // TODO Remove if remains Unused || Ask if this Ip is subject to change?
        /** @brief Name of the network adapter this camera is connected to  */
        std::string interfaceName;
        uint32_t interfaceIndex = 0;
        /** @brief Flag for registering if device is clicked in sidebar */
        bool clicked;
        /** @brief Current connection state for this device */
        ArConnectionState state;

        std::map<int, StreamingModes> streams;

        PreviewLayout layout = PREVIEW_LAYOUT_NONE;
        CameraPlaybackFlags playbackStatus = AR_PREVIEW_NONE;
        std::vector<std::string> modes;
        uint32_t selectedModeIndex = 0;
        uint32_t selectedSourceIndex = 0;
        CRLCameraResolution selectedMode;
        std::vector<std::string> sources; // Human-readable names of camera sources
        std::map<uint32_t, std::string> selectedSourceMap;
        std::map<uint32_t, uint32_t> selectedSourceIndexMap;
        std::map<uint32_t, int> hoveredPixelInfo;

        std::vector<std::string> userRequestedSources;
        std::vector<std::string> enabledStreams;
        std::vector<std::string> attachedScripts;
        float row[9] = {0};
        float col[9] = {0};
        bool pixelInfoEnable = false;
        CursorPixelInformation pixelInfo;
        bool useLoadedProfile = false;

        /** @brief object containing all adjustable parameters to the camera */
        Parameters parameters{};

        Page selectedPreviewTab = TAB_2D_PREVIEW;
        /** @brief  Showing point cloud view*/
        bool pointCloud = false;
        /** @brief  Showing depth image stream*/
        bool depthImage = false;
        /** @brief  Showing color image stream*/
        bool colorImage = false;

        /** @brief Show a default preview with some selected streams*/
        bool button = false;
    };


    struct EntryConnectDevice {
        std::string profileName = "MultiSense";
        std::string IP = "10.66.171.21";
        std::string interfaceName;
        uint32_t interfaceIndex{};

        std::string cameraName;

        EntryConnectDevice() = default;

        EntryConnectDevice(std::string ip, std::string iName, std::string camera, uint32_t idx) : IP(std::move(ip)),
                                                                                                  interfaceName(
                                                                                                          std::move(
                                                                                                                  iName)),
                                                                                                  cameraName(std::move(
                                                                                                          camera)),
                                                                                                  interfaceIndex(idx) {
        }

        void reset() {
            profileName = "";
            IP = "";
            interfaceName = "";
            interfaceIndex = 0;
        }

        bool ready(const std::vector<AR::Element> *devices, const EntryConnectDevice &entry) const {

            bool profileNameEmpty = entry.profileName.empty();
            bool profileNameTaken = false;
            bool IPEmpty = entry.IP.empty();
            bool adapterNameEmpty = entry.interfaceName.empty();

            bool AdapterAndIPInTaken = false;



            // Loop through devices and check that it doesn't exist already.
            for (auto &d: *devices) {
                if (d.IP == entry.IP && d.interfaceName == entry.interfaceName) {
                    AdapterAndIPInTaken = true;
                    Log::Logger::getInstance()->info("Ip {} on adapter {} already in use", entry.IP,
                                                     entry.interfaceName);

                }
                if (d.name == entry.profileName) {
                    profileNameTaken = true;
                    Log::Logger::getInstance()->info("Profile name '{}' already taken", entry.profileName);
                }

            }

            bool ready = true;
            if (profileNameEmpty || profileNameTaken || IPEmpty || adapterNameEmpty || AdapterAndIPInTaken)
                ready = false;

            return ready;
        }
    };

}
namespace ArEngine {
    struct YUVTexture {
        void *data[NUM_YUV_DATA_POINTERS]{};
        uint32_t len[NUM_YUV_DATA_POINTERS] = {0};
        VkFlags *formats[NUM_YUV_DATA_POINTERS]{};
        VkFormat format{};
    };

    struct TextureData {
        TextureData(CRLCameraDataType texType) : type(texType) {

        }
        TextureData() {
        }

        void *data = nullptr;
        uint32_t len = 0;
        CRLCameraDataType type = AR_CAMERA_IMAGE_NONE;

        struct {
            void *data[NUM_YUV_DATA_POINTERS]{};
            uint32_t len[NUM_YUV_DATA_POINTERS] = {0};
            VkFlags *formats[NUM_YUV_DATA_POINTERS]{};
            VkFormat format{};
        } planar;
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


    struct UBOMatrix {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
    };

    struct FragShaderParams {
        glm::vec4 lightColor;
        glm::vec4 objectColor;
        glm::vec4 lightPos;
        glm::vec4 viewPos;
    };

    struct PointCloudParam {
        glm::mat4 kInverse;
        float width;
        float height;
    };

    struct ZoomParam {
        float zoom;

        ZoomParam() {
            zoom = 1.0f;
        }
    };

    struct PointCloudShader {
        glm::vec4 pos[NUM_POINTS];
        glm::vec4 col[NUM_POINTS];
    };

    struct MousePositionPushConstant {
        glm::vec2 position;
    };

    struct ObjectPicking {
        // Global render pass for frame buffer writes
        VkRenderPass renderPass;
        // List of available frame buffers (same as number of swap chain images)
        VkFramebuffer frameBuffer;

        VkImage colorImage;
        VkImage depthImage;
        VkImageView colorView;
        VkImageView depthView;
        VkDeviceMemory colorMem;
        VkDeviceMemory depthMem;

    };

}


#endif //MULTISENSE_DEFINITIONS_H
