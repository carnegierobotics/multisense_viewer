/**
 * @file: MultiSense-Viewer/include/Viewer/Core/Definitions.h
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
 *   2021-05-15, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_DEFINITIONS_H
#define MULTISENSE_DEFINITIONS_H

//#define MULTISENSE_VIEWER_PRODUCTION // Disable validation layers and other test functionality

#ifndef MULTISENSE_VIEWER_PRODUCTION
    #define MULTISENSE_VIEWER_DEBUG
#endif

#include <unordered_map>
#include <memory>
#include <utility>
#include <array>
#include <vulkan/vulkan_core.h>
#include <MultiSense/MultiSenseTypes.hh>
#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>

#define IMGUI_DEFINE_MATH_OPERATORS

#include <imgui.h>

#include "Viewer/Core/KeyInput.h"
#include "Viewer/Core/Buffer.h"
#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Camera.h"
#include "Viewer/Core/Texture.h"
#include "Viewer/Tools/ThreadPool.h"

#define INTERVAL_10_SECONDS 10
#define INTERVAL_1_SECOND 1
#define INTERVAL_2_SECONDS 2
#define INTERVAL_5_SECONDS 5
#define MAX_IMAGES_IN_QUEUE 5




// Predeclare to speed up compile times
namespace VkRender::MultiSense {
    class CRLPhysicalCamera;
}


namespace Log {
    // enum for LOG_LEVEL
    typedef enum LOG_LEVEL {
        DISABLE_LOG = 1,
        LOG_LEVEL_INFO = 2,
        LOG_LEVEL_BUFFER = 3,
        LOG_LEVEL_TRACE = 4,
        LOG_LEVEL_DEBUG = 5,
        ENABLE_LOG = 6,
    } LogLevel;


    // enum for LOG_TYPE
    typedef enum LOG_TYPE {
        NO_LOG = 1,
        CONSOLE = 2,
        FILE_LOG = 3,
    } LogType;
}

/**
 * @brief Defines draw behaviour of a script
 */
typedef enum ScriptType {
    /** CRL_SCRIPT_TYPE_DISABLED Do not draw script at all */
    CRL_SCRIPT_TYPE_DISABLED,
    /** CRL_SCRIPT_TYPE_ADDITIONAL_BUFFERS Draw script since first frame and allocate additional MVP buffers */
    CRL_SCRIPT_TYPE_ADDITIONAL_BUFFERS,
    /** CRL_SCRIPT_TYPE_DEFAULT Draw script after crl camera connect */
    CRL_SCRIPT_TYPE_DEFAULT,
    /** CRL_SCRIPT_TYPE_RENDER Draw script since application startup. No particular order */
    CRL_SCRIPT_TYPE_RENDER,
    /** CRL_SCRIPT_TYPE_RELOAD This script is set to reload (destroy and create) next frame */
    CRL_SCRIPT_TYPE_RELOAD,
    /**
     * Create this script before default and always render this type first. No internal ordering amongst scripts
     */
    CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE,
    CRL_SCRIPT_TYPE_RENDER_PBR,
} ScriptType;

/**
 * @brief Labels data coming from the camera to a type used to initialize textures with various formats and samplers
 */
typedef enum CRLCameraDataType {
    CRL_DATA_NONE,
    CRL_GRAYSCALE_IMAGE,
    CRL_COLOR_IMAGE_RGBA,
    CRL_COLOR_IMAGE_YUV420,
    CRL_CAMERA_IMAGE_NONE,
    CRL_DISPARITY_IMAGE,
    CRL_POINT_CLOUD
} CRLCameraDataType;

/**
 * @brief What connection state a device seen in the side bar can be in.
 */
typedef enum ArConnectionState {
    CRL_STATE_CONNECTED = 0,                /// Not implemented cause its the same state as ACTIVE
    CRL_STATE_CONNECTING = 1,               /// Just clicked connect
    CRL_STATE_ACTIVE = 2,                   /// Normal operation
    CRL_STATE_INACTIVE = 3,                 /// Not sure ?
    CRL_STATE_LOST_CONNECTION = 4,          /// Lost connection... Will go into UNAVAILABLE state
    CRL_STATE_RESET = 5,                    /// Reset this device (Put it in disconnected state) if it was clicked while being in ACTIVE state or another device was activated
    CRL_STATE_DISCONNECTED = 6,             /// Not currently active but can be activated by clicking it
    CRL_STATE_UNAVAILABLE = 7,              /// Could not be found on any network adapters
    CRL_STATE_DISCONNECT_AND_FORGET = 8,    /// Device is removed and the profile in crl.ini is deleted
    CRL_STATE_REMOVE_FROM_LIST = 9,         /// Device is removed from sidebar list on next frame
    CRL_STATE_INTERRUPT_CONNECTION = 10,    /// Stop attempting to connect (For closing application quickly and gracefully)
    CRL_STATE_JUST_ADDED = 11,               /// Skip to connection flow, just added into sidebar list
    CRL_STATE_JUST_ADDED_WINDOWS = 12        /// Windows autoconnect needs some extra time to propogate
} ArConnectionState;

/**
 * @brief Identifier for each preview window and point cloud type. Loosely tied to the preview scripts
 * Note: Numbering matters
 */
typedef enum StreamWindowIndex {
    // First 0 - 8 elements also correspond to array indices. Check upon this before adding more PREVIEW indices
    CRL_PREVIEW_ONE = 0,
    CRL_PREVIEW_TWO = 1,
    CRL_PREVIEW_THREE = 2,
    CRL_PREVIEW_FOUR = 3,
    CRL_PREVIEW_FIVE = 4,
    CRL_PREVIEW_SIX = 5,
    CRL_PREVIEW_SEVEN = 6,
    CRL_PREVIEW_EIGHT = 7,
    CRL_PREVIEW_NINE = 8,
    CRL_PREVIEW_POINT_CLOUD = 9,
    CRL_PREVIEW_TOTAL_MODES = CRL_PREVIEW_POINT_CLOUD + 1,
    // Other flags
} StreamWindowIndex;


/**
 * @brief Identifier for different pages in the GUI.
 */
typedef enum page {
    CRL_PAGE_PREVIEW_DEVICES = 0,
    CRL_PAGE_DEVICE_INFORMATION = 1,
    CRL_PAGE_CONFIGURE_DEVICE = 2,
    CRL_PAGE_TOTAL_PAGES = 3,
    CRL_TAB_NONE = 10,
    CRL_TAB_2D_PREVIEW = 11,
    CRL_TAB_3D_POINT_CLOUD = 12,

    CRL_TAB_PREVIEW_CONTROL = 20,
    CRL_TAB_SENSOR_CONFIG = 21
} Page;

/**
 * @brief Hardcoded Camera resolutions using descriptive enums
 */
typedef enum CRLCameraResolution {
    CRL_RESOLUTION_960_600_64 = 0,
    CRL_RESOLUTION_960_600_128 = 1,
    CRL_RESOLUTION_960_600_256 = 2,
    CRL_RESOLUTION_1920_1200_64 = 3,
    CRL_RESOLUTION_1920_1200_128 = 4,
    CRL_RESOLUTION_1920_1200_256 = 5,
    CRL_RESOLUTION_1024_1024_128 = 6,
    CRL_RESOLUTION_2048_1088_256 = 7,
    CRL_RESOLUTION_NONE = 8
} CRLCameraResolution;

/**
 * @brief Page within pages. Which layout is chosen for previews
 */
typedef enum PreviewLayout {
    CRL_PREVIEW_LAYOUT_NONE = 0,
    CRL_PREVIEW_LAYOUT_SINGLE = 1,
    CRL_PREVIEW_LAYOUT_DOUBLE = 2,
    CRL_PREVIEW_LAYOUT_DOUBLE_SIDE_BY_SIDE = 3,
    CRL_PREVIEW_LAYOUT_QUAD = 4,
    CRL_PREVIEW_LAYOUT_NINE = 5
} PreviewLayout;

typedef enum ScriptWidgetType {
    WIDGET_FLOAT_SLIDER = 0,
    WIDGET_INT_SLIDER = 1,
    WIDGET_INPUT_NUMBER = 2,
    WIDGET_TEXT = 3,
    WIDGET_CHECKBOX = 4
} ScriptWidgetType;

/**
 * @brief Adjustable sensor parameters
 */
struct LightingParams {
    float dutyCycle = 1.0f;
    int selection = -1;
    bool flashing = true;
    float numLightPulses = 3;
    float startupTime = 0;
    bool update = false;
};

/**
 * @brief Adjustable sensor parameters
 */
struct ExposureParams {
    bool autoExposure{};
    int exposure = 20000;
    uint32_t autoExposureMax = 20000;
    uint32_t autoExposureDecay = 7;
    float autoExposureTargetIntensity = 0.5f;
    float autoExposureThresh = 0.9f;
    uint16_t autoExposureRoiX = 0;
    uint16_t autoExposureRoiY = 0;
    uint16_t autoExposureRoiWidth = crl::multisense::Roi_Full_Image;
    uint16_t autoExposureRoiHeight = crl::multisense::Roi_Full_Image;
    uint32_t currentExposure = 0;

    bool update = false;
};

struct AUXConfig {
    float gain = 1.7f;
    float whiteBalanceBlue = 1.0f;
    float whiteBalanceRed = 1.0f;
    bool whiteBalanceAuto = true;
    uint32_t whiteBalanceDecay = 3;
    float whiteBalanceThreshold = 0.5f;
    bool hdr = false;
    float gamma = 2.0f;
    bool sharpening = false;
    float sharpeningPercentage = 50.0f;
    int sharpeningLimit = 50;

    ExposureParams ep{};
    bool update = false;
};

struct ImageConfig {
    float gain = 1.0f;
    float fps = 30.0f;
    float stereoPostFilterStrength = 0.5f;
    bool hdrEnabled = false;
    float gamma = 2.0f;
    bool hdr = false;

    ExposureParams ep{};
    bool update = false;

};

/**
 * @brief Adjustable sensor parameters
 */
struct CalibrationParams {
    bool update = false;
    bool save = false;
    std::string intrinsicsFilePath = "Path/To/Intrinsics.yml";
    std::string extrinsicsFilePath = "Path/To/Extrinsics.yml";
    std::string saveCalibrationPath = "Path/To/Dir";
    bool updateFailed = false;
    bool saveFailed = false;

    CalibrationParams() {
        intrinsicsFilePath.resize(255);
        extrinsicsFilePath.resize(255);
        saveCalibrationPath.resize(255);
    }
};

/** @brief  MultiSense Device configurable parameters. Should contain all adjustable parameters through LibMultiSense */
struct Parameters {
    LightingParams light{};
    CalibrationParams calib{};
    ImageConfig stereo;
    AUXConfig aux;

    bool updateGuiParams = true;
};

/**
 * @brief MAIN RENDER NAMESPACE. This namespace contains all Render resources presented by the backend of this renderer engine.
 */
namespace VkRender {
    /**
     * @brief GLFW and Vulkan combination to create a SwapChain
     */
    typedef struct SwapChainCreateInfo {
        GLFWwindow *pWindow{};
        bool vsync = true;
        VkInstance instance{};
        VkPhysicalDevice physicalDevice{};
        VkDevice device{};
    } SwapChainCreateInfo;

    /**
     * @brief Data block for displaying pixel intensity and depth in textures on mouse hover
     */
    struct CursorPixelInformation {
        uint32_t x = 0, y = 0;
        uint32_t r = 0, g = 0, b = 0;
        uint32_t intensity = 0;
        float depth = 0;
    };


    /**
     * Shared data for image effects
     */
    struct ImageEffectData {
        float minDisparityValue = 0.0f;
        float maxDisparityValue = 255.0f;

    };

    /**
    * Image effect options for each preview window
    */
    struct ImageEffectOptions {
        bool normalize = false;
        bool interpolation = false;
        bool depthColorMap = false;
        bool magnifyZoomMode = false;
        bool edgeDetection = false;
        bool blur = false;
        bool emboss = false;
        bool sharpening = false;
        VkRender::ImageEffectData data;
    };


    /**
     * @brief Information block for each Preview window, should support 1-9 possible previews. Contains stream names and info to presented to the user.
     * Contains a mix of GUI related and MultiSense Device related info.
     */
    struct PreviewWindow {
        std::vector<std::string> availableSources{}; // Human-readable names of camera sources
        std::vector<std::string> availableRemoteHeads{};
        std::string selectedSource = "Idle";
        uint32_t selectedSourceIndex = 0;
        crl::multisense::RemoteHeadChannel selectedRemoteHeadIndex = 0;
        float xPixelStartPos = 0;
        float yPixelStartPos = 0;
        float row = 0;
        float col = 0;

        VkRender::ImageEffectOptions effects;
        bool enableZoom = true;
        bool isHovered = false;
        ImVec2 popupPosition = ImVec2(0.0f,
                                      0.0f); // Position of popup window for image effects. (Used to update popup position when preview window is scrolled)
        ImVec2 popupWindowSize = ImVec2(0.0f,
                                        0.0f); // Position of popup window for image effects. (Used to update popup position when preview window is scrolled)
        bool updatePosition = false;
        std::string name = "None";
    };

    /**
     * @brief Ties together the preview windows for each MultiSense channel
     * Contain possible streams and resolutions for each channel
     */
    struct ChannelInfo {
        uint32_t index = 0;
        std::vector<std::string> availableSources{};
        /** @brief Current connection state for this channel */
        ArConnectionState state = CRL_STATE_DISCONNECTED;
        std::vector<std::string> modes{};
        uint32_t selectedModeIndex = 0;
        CRLCameraResolution selectedResolutionMode = CRL_RESOLUTION_NONE;
        std::vector<std::string> requestedStreams{};
        std::vector<std::string> enabledStreams{};
        bool updateResolutionMode = true;
    };

    /**
     * @brief UI Block for a MultiSense Device connection. Contains connection information and user configuration such as selected previews, recording info and more..
     */
    struct Device {
        /** @brief Profile Name information  */
        std::string name = "Profile #1";
        /** @brief Name of the camera connected  */
        std::string cameraName;
        /** @brief Identifier of the camera connected  */
        std::string serialName;
        /** @brief IP of the connected camera  */
        std::string IP;
        /** @brief Default IP of the connected camera  */
        std::string defaultIP = "10.66.171.21";
        /** @brief Name of the network adapter this camera is connected to  */
        std::string interfaceName;
        /** @brief InterFace index of network adapter */
        uint32_t interfaceIndex = 0;
        /** @brief Descriptive text of interface if provided by the OS (Required on windows) */
        std::string interfaceDescription;
        /** @brief Flag for registering if device is clicked in sidebar */
        bool clicked = false;
        /** @brief Current connection state for this profile */
        ArConnectionState state = CRL_STATE_UNAVAILABLE;
        /** @brief is this device a remote head or a MultiSense camera */
        bool isRemoteHead = false;
        /** @brief Order each preview window with a index flag*/
        std::unordered_map<StreamWindowIndex, PreviewWindow> win{};
        /** @brief Put each camera interface channel (crl::multisense::Channel) 0-3 in a vector. Contains GUI info for each channel */
        std::vector<ChannelInfo> channelInfo{};
        /** @brief object containing all adjustable parameters to the camera */
        Parameters parameters{};
        /**@brief location for which this m_Device should save recorded frames **/
        std::string outputSaveFolder;
        /**@brief location for which this m_Device should save recorded point clouds **/
        std::string outputSaveFolderPointCloud;
        /**@brief location for which this m_Device should save recorded point clouds **/
        std::string outputSaveFolderIMUData;
        /**@brief Flag to decide if user is currently recording frames */
        bool isRecording = false;
        /**@brief Flag to decide if user is currently recording point cloud */
        bool isRecordingPointCloud = false;
        /**@brief Flag to decide if user is currently recording IMU data */
        bool isRecordingIMUdata = false;
        /** @brief 3D view camera type for this device. Arcball or first person view controls) */
        int cameraType = 0;
        /** @brief Reset 3D view camera position and rotation */
        bool resetCamera = false;
        /** @brief Pixel information from renderer, on mouse hover for textures */
        std::unordered_map<StreamWindowIndex, CursorPixelInformation> pixelInfo{};
        /** @brief Pixel information scaled after zoom */
        std::unordered_map<StreamWindowIndex, CursorPixelInformation> pixelInfoZoomed{};
        /** @brief Indices for each remote head if connected.*/
        std::vector<crl::multisense::RemoteHeadChannel> channelConnections{};
        /** @brief Config index for remote head. Presented as radio buttons for remote head selection in the GUI. 0 is for MultiSense */
        crl::multisense::RemoteHeadChannel configRemoteHead = 0;
        /** @brief Flag to signal application to revert network settings on application exit */
        bool systemNetworkChanged = false;
        /** Interrupt connection flow if users exits program. */
        bool interruptConnection = false;
        /** @brief Which compression method to use. Default is tiff which offer no compression */
        std::string saveImageCompressionMethod = "tiff";
        /** @brief Not a physical device just testing the GUI */
        bool notRealDevice = false;
        /** @brief If possible then use the IMU in the camera */
        bool useIMU = true;
        /** @brief 0 : luma // 1 : Color  */
        int useAuxForPointCloudColor = 1;

        /** @brief Following is UI elements settings for the active device **/
        Page controlTabActive = CRL_TAB_PREVIEW_CONTROL;
        /** @brief Which TAB this preview has selected. 2D or 3D view. */
        Page selectedPreviewTab = CRL_TAB_2D_PREVIEW;
        /** @brief What type of layout is selected for this device*/
        PreviewLayout layout = CRL_PREVIEW_LAYOUT_SINGLE;
        /** @brief IF 3D area should be extended or not */
        bool extend3DArea = true;
        /** @brief If the connected device has a color camera */
        bool hasColorCamera = false;
        /** @brief If we managed to update all the device configs */
        bool updateDeviceConfigsSucceeded = false;
        Device() {
            outputSaveFolder.resize(255);
            outputSaveFolderPointCloud.resize(255);
        }
    };

    /**
     * @brief Default Vertex information
     */
    struct Vertex {
        glm::vec3 pos{};
        glm::vec3 normal{};
        glm::vec2 uv0{};
        glm::vec2 uv1{};
        glm::vec4 joint0{};
        glm::vec4 weight0{};
        glm::vec4 color{};
    };

    /**
     * @brief MouseButtons user input
     */
    struct MouseButtons {
        bool left = false;
        bool right = false;
        bool middle = false;
        int action = 0;
        float wheel = 0.0f; // to initialize arcball zoom
        float dx = 0.0f;
        float dy = 0.0f;
        struct {
            float x = 0.0f;
            float y = 0.0f;
        } pos;
    };

    /**
     * @brief Default MVP matrices
     */
    struct UBOMatrix {
        glm::mat4 projection{};
        glm::mat4 view{};
        glm::mat4 model{};
        glm::vec3 camPos;
    };

    /**
     * @brief Basic lighting params for simple light calculation
     */
    struct FragShaderParams {
        glm::vec4 lightDir{};
        glm::vec4 zoomCenter{};
        glm::vec4 zoomTranslate{};
        float exposure = 4.5f;
        float gamma = 2.2f;
        float prefilteredCubeMipLevels = 0.0f;
        float scaleIBLAmbient = 1.0f;
        float debugViewInputs = 0.0f;
        float lod = 0.0f;
        glm::vec2 pad{};
        glm::vec4 disparityNormalizer; // (0: should normalize?, 1: min value, 2: max value, 3 pad)
        glm::vec4 kernelFilters; // 0 Sobel/Edge kernel, Blur kernel,
    };

    struct SkyboxTextures {
        TextureCubeMap environmentMap{};
        TextureCubeMap irradianceCube{};
        TextureCubeMap prefilterEnv{};
        Texture2D lutBrdf{};
        float prefilteredCubeMipLevels = 0;
    };

    /**
     * @brief Memory block for point clouds
     */
    struct PointCloudParam {
        /** @brief Q Matrix. See PointCloudUtility in LibMultiSense for calculation of this */
        glm::mat4 Q{};
        /** @brief Width of depth image*/
        float width{};
        /** @brief Height of depth image*/
        float height{};
        /** @brief Max disparity of image*/
        float disparity{};
        /** @brief Distance between left and right camera (tx)*/
        float focalLength{};
        /** @brief Scaling factor used when operating in cropped mode assuming uniform scaling in x- and y direction */
        float scale{};
        /** @brief Point size to view the point cloud. Larger for more distant points and smaller for closer points */
        float pointSize{};
    };

    struct ColorPointCloudParams {
        glm::mat4 instrinsics{};
        glm::mat4 extrinsics{};
        float useColor = true;
        float hasSampler = false;
    };

    /** @brief Additional default buffers for rendering mulitple models with distrinct MVP */
    struct RenderDescriptorBuffers {
        UBOMatrix mvp{};
        FragShaderParams light{};
    };

    /**@brief A standard set of uniform buffers. All current shaders can get away with using a combination of these two */
    struct RenderDescriptorBuffersData {
        Buffer mvp{};
        Buffer light{};
    };
    /**
     * @brief TODO: Unfinished. Put mouse cursor position information into shader. Meant for zoom feature
     */
    struct MousePositionPushConstant {
        glm::vec2 position{};
    };


    /**
     * @brief Vulkan resources for the cursor information pipeline
     */
    struct ObjectPicking {
        // Global render pass for frame buffer writes
        VkRenderPass renderPass{};
        // List of available frame buffers (same as number of swap chain images)
        VkFramebuffer frameBuffer{};
        VkImage colorImage{};
        VkImage depthImage{};
        VkImageView colorView{};
        VkImageView depthView{};
        VkDeviceMemory colorMem{};
        VkDeviceMemory depthMem{};
    };


    /**@brief A standard set of uniform buffers. Each script is initialized with these */
    struct UniformBufferSet {
        Buffer bufferOne{};
        Buffer bufferTwo{};
        Buffer bufferThree{};
    };

    /** Containing Basic Vulkan Resources for rendering for use in scripts **/
    struct RenderUtils {
        VulkanDevice *device{};
        uint32_t UBCount = 0;
        VkRenderPass *renderPass{};
        std::vector<UniformBufferSet> uniformBuffers{};
        const VkRender::ObjectPicking *picking = nullptr;
        struct {
            TextureCubeMap *irradianceCube;
            TextureCubeMap *prefilterEnv;
            Texture2D *lutBrdf;
            float prefilteredCubeMipLevels = 0.0f;
        } skybox;
        std::mutex *queueSubmitMutex;
    };

    /**@brief grouping containing useful pointers used to render scripts. This will probably change frequently as the viewer grows **/
    struct RenderData {

        uint32_t index = 0;
        Camera *camera = nullptr;
        float deltaT = 0.0f;
        bool drawThisScript = false;
        /**
         * @brief Runtime measured in seconds
         */
        float scriptRuntime = 0.0f;
        int scriptDrawCount = 0;
        std::string scriptName;
        MultiSense::CRLPhysicalCamera *crlCamera{};
        ScriptType type{};
        uint32_t height = 0;
        uint32_t width = 0;
        const Input *input = nullptr;
        bool additionalBuffers = false;
    };

}


#endif //MULTISENSE_DEFINITIONS_H
