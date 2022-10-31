//
// Created by magnus on 15/05/21.
//

#ifndef MULTISENSE_DEFINITIONS_H
#define MULTISENSE_DEFINITIONS_H


#include "glm/glm.hpp"
#include "vulkan/vulkan_core.h"
#include "include/MultiSense/MultiSenseTypes.hh"
#include "unordered_map"
#include "memory"
#include "GLFW/glfw3.h"
#include "Buffer.h"
#include "VulkanDevice.h"
#include "Camera.h"
#include "KeyInput.h"
#include "imgui.h"
#include <utility>
#include <array>

#define NUM_YUV_DATA_POINTERS 3
#define NUM_POINTS 2048 // Changing this also needs to be changed in the vs shader.
#define INTERVAL_10_SECONDS 10
#define INTERVAL_1_SECOND 1
#define STACK_SIZE_100 100


namespace VkRender::MultiSense {
    class CRLPhysicalCamera;
}

typedef enum ScriptType {
    AR_SCRIPT_TYPE_DISABLED,
    AR_SCRIPT_TYPE_DEFAULT,
    AR_SCRIPT_TYPE_CRL_CAMERA,
    AR_SCRIPT_TYPE_POINT_CLOUD,
    AR_SCRIPT_TYPE_LAYOUT_SINGLE,
    AR_SCRIPT_TYPE_LAYOUT_DOUBLE_TOP,
    AR_SCRIPT_TYPE_LAYOUT_DOUBLE_BOT,
    AR_SCRIPT_TYPE_LAYOUT_QUAD_ONE
} ScriptType;


typedef enum CRLCameraDataType {
    AR_POINT_CLOUD,
    AR_GRAYSCALE_IMAGE,
    AR_COLOR_IMAGE,
    AR_COLOR_IMAGE_YUV420,
    AR_YUV_PLANAR_FRAME,
    AR_CAMERA_IMAGE_NONE,
    AR_DISPARITY_IMAGE
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
    AR_STATE_LOST_CONNECTION = 4,
    AR_STATE_RESET = 5,
    AR_STATE_DISCONNECTED = 6,
    AR_STATE_UNAVAILABLE = 7,
    AR_STATE_DISCONNECT_AND_FORGET = 8,
    AR_STATE_REMOVE_FROM_LIST = 9,
    AR_STATE_JUST_ADDED = 10
} ArConnectionState;


typedef enum StreamIndex {
    // First 0 - 8 elements also correspond to array indices. Check upon this before adding more PREVIEW indices
    AR_PREVIEW_ONE = 0,
    AR_PREVIEW_TWO = 1,
    AR_PREVIEW_THREE = 2,
    AR_PREVIEW_FOUR = 3,
    AR_PREVIEW_FIVE = 4,
    AR_PREVIEW_SIX = 5,
    AR_PREVIEW_SEVEN = 6,
    AR_PREVIEW_EIGHT = 7,
    AR_PREVIEW_NINE = 8,
    AR_PREVIEW_POINT_CLOUD = 9,


    AR_PREVIEW_TOTAL_MODES = AR_PREVIEW_POINT_CLOUD + 1,
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

typedef enum CRLCameraBaseUnit {
    CRL_BASE_MULTISENSE = 0,
    CRL_BASE_REMOTE_HEAD = 1
} CRLCameraBaseUnit;

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
    bool update = false;
};

struct LightingParams {
    float dutyCycle = 1.0f;
    int selection = -1;
    bool flashing = true;
    uint32_t numLightPulses = 3;
    uint32_t startupTime = 0;
    bool update = false;
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

    bool update = false;
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
        bool vsync = false;
        VkInstance instance{};
        VkPhysicalDevice physicalDevice{};
        VkDevice device{};
    } SwapChainCreateInfo;

    /** @brief  MultiSense Device configurable parameters. Should contain all adjustable parameters through LibMultiSense */
    struct Parameters {
        ExposureParams ep{};
        WhiteBalanceParams wb{};
        LightingParams light{};

        float gain = 1.0f;
        float fps = 30.0f;
        float stereoPostFilterStrength = 0.5f;
        bool hdrEnabled = false;
        float gamma = 2.0f;

        bool update = false;
        bool updateGuiParams = true;
    };

    struct CursorPixelInformation {
        uint32_t x{}, y{};
        uint32_t r{}, g{}, b{};
        uint32_t intensity{};
        uint32_t depth{};
    };

    struct PreviewWindow {
        std::vector<std::string> availableSources{}; // Human-readable names of camera sources
        std::vector<std::string> availableRemoteHeads{};
        std::string selectedSource = "Source";
        uint32_t selectedSourceIndex = 0;
        int hoveredPixelInfo{};
        crl::multisense::RemoteHeadChannel selectedRemoteHeadIndex = 0;
    };

    struct ChannelInfo {
        uint32_t index = 0;
        std::vector<std::string> availableSources{}; // Human-readable names of camera sources
        ArConnectionState state = AR_STATE_DISCONNECTED;
        std::vector<std::string> modes{};
        uint32_t selectedModeIndex = 0;
        CRLCameraResolution selectedMode{};
        std::vector<std::string> requestedStreams{};
        std::vector<std::string> enabledStreams{};
        bool updateResolutionMode = true;
    };

    /**
     * @brief UI Block for a MultiSense Device connection. Contains connection information and user configuration such as selected previews, recording info and more..
     */
    struct Device {
        /** @brief Profile Name information  */
        std::string name = "Profile #1"; // TODO Remove if remains Unused
        /** @brief Name of the camera connected  */
        std::string cameraName;
        /** @brief Identifier of the camera connected  */
        std::string serialName;
        /** @brief IP of the connected camera  */
        std::string IP;
        /** @brief Default IP of the connected camera  */
        std::string defaultIP = "10.66.171.21"; // TODO Remove if remains Unused || Ask if this Ip is subject to change?
        /** @brief Name of the network adapter this camera is connected to  */
        std::string interfaceName;
        uint32_t interfaceIndex = 0;
        std::string interfaceDescription;
        /** @brief Flag for registering if device is clicked in sidebar */
        bool clicked = false;
        /** @brief Current connection state for this device */
        ArConnectionState state = AR_STATE_UNAVAILABLE;
        CameraPlaybackFlags playbackStatus = AR_PREVIEW_NONE;
        PreviewLayout layout = PREVIEW_LAYOUT_NONE;

        std::unordered_map<uint32_t, PreviewWindow> win{};
        std::vector<ChannelInfo> channelInfo{};
        CRLCameraBaseUnit baseUnit{};

        /**@brief location for which this device should save recorded frames **/
        std::string outputSaveFolder = "/Path/To/Folder/";
        bool isRecording = false;

        float row[9] = {0};
        float col[9] = {0};
        bool pixelInfoEnable = false;
        bool useImuData = false;
        CursorPixelInformation pixelInfo{};
        std::vector<crl::multisense::RemoteHeadChannel> channelConnections{};
        int configRemoteHead = 0;
        /** @brief object containing all adjustable parameters to the camera */
        Parameters parameters{};

        Page selectedPreviewTab = TAB_2D_PREVIEW;
        /** @brief Show a default preview with some selected streams*/
        bool systemNetworkChanged = false;
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
    };

    struct Rotation {
        glm::mat4 rotationMatrix{};
        float roll = 0;
        float pitch = 0;
        float yaw = 0;
    };

    struct MouseButtons {
        bool left = false;
        bool right = false;
        bool middle = false;
        float wheel = 0.0f;
    };

    /**
     * @brief Default MVP matrix
     */
    struct UBOMatrix {
        glm::mat4 projection{};
        glm::mat4 view{};
        glm::mat4 model{};
    };

    /**
     * @brief Basic lighting params for simple light calculation
     */
    struct FragShaderParams {
        glm::vec4 lightColor{};
        glm::vec4 objectColor{};
        glm::vec4 lightPos{};
        glm::vec4 viewPos{};
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
    };

    /**
     * @brief TODO: Unfinished. Put zoom information into shader. Meant for zooming
     */
    struct ZoomParam {
        float zoom{};

        ZoomParam() {
            zoom = 1.0f;
        }
    };

    /**
     * @brief TODO: Unfinished. Put mouse cursor position information into shader. Meant for zooming
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


    /**@brief A standard set of uniform buffers. All current shaders can get away with using a combination of these four */
    struct UniformBufferSet {
        Buffer bufferOne{};
        Buffer bufferTwo{};
        Buffer bufferThree{};
        Buffer bufferFour{};
    };

    /** containing Basic Vulkan Resources for rendering **/
    struct RenderUtils {
        VulkanDevice *device{};
        uint32_t UBCount = 0;
        VkRenderPass *renderPass{};
        std::vector<UniformBufferSet> uniformBuffers{};
        const VkRender::ObjectPicking *picking = nullptr;
    };

    /**@brief grouping containing useful pointers used to render scripts. This will probably change frequently as the viewer grows **/
    struct RenderData {
        uint32_t index = 0;
        Camera *camera = nullptr;
        float deltaT = 0.0f;
        bool drawThisScript = false;
        float scriptRuntime = 0.0f;
        int scriptDrawCount = 0;
        std::string scriptName;
        std::unique_ptr<MultiSense::CRLPhysicalCamera> *crlCamera{};
        ScriptType type{};
        uint32_t height = 0;
        uint32_t width = 0;
        const Input *input = nullptr;
    };


    static const ImVec4 yellow(0.98f, 0.65f, 0.00f, 1.0f);
    static const ImVec4 green(0.26f, 0.42f, 0.31f, 1.0f);
    static const ImVec4 TextGreenColor(0.16f, 0.95f, 0.11f, 1.0f);
    static const ImVec4 TextRedColor(0.95f, 0.045f, 0.041f, 1.0f);
    static const ImVec4 red(0.613f, 0.045f, 0.046f, 1.0f);
    static const ImVec4 DarkGray(0.1f, 0.1f, 0.1f, 1.0f);
    static const ImVec4 PopupTextInputBackground(0.01f, 0.05f, 0.1f, 1.0f);
    static const ImVec4 TextColorGray(0.75f, 0.75f, 0.75f, 1.0f);
    static const ImVec4 PopupHeaderBackgroundColor(0.15f, 0.25, 0.4f, 1.0f);
    static const ImVec4 PopupBackground(0.183f, 0.33f, 0.47f, 1.0f);
    static const ImVec4 CRLGray421(0.666f, 0.674f, 0.658f, 1.0f);
    static const ImVec4 CRLGray424(0.411f, 0.419f, 0.407f, 1.0f);
    static const ImVec4 CRLCoolGray(0.870f, 0.878f, 0.862f, 1.0f);
    static const ImVec4 CRLCoolGrayTransparent(0.870f, 0.878f, 0.862f, 0.5f);
    static const ImVec4 CRLGray424Main(0.462f, 0.474f, 0.494f, 1.0f);
    static const ImVec4 CRLDarkGray425(0.301f, 0.313f, 0.309f, 1.0f);
    static const ImVec4 CRLRed(0.768f, 0.125f, 0.203f, 1.0f);
    static const ImVec4 CRLRedHover(0.86f, 0.378f, 0.407f, 1.0f);
    static const ImVec4 CRLRedActive(0.96f, 0.478f, 0.537f, 1.0f);
    static const ImVec4 CRLBlueIsh(0.313f, 0.415f, 0.474f, 1.0f);
    static const ImVec4 CRLBlueIshTransparent(0.313f, 0.415f, 0.474f, 0.3f);
    static const ImVec4 CRLBlueIshTransparent2(0.313f, 0.415f, 0.474f, 0.1f);
    static const ImVec4 CRLTextGray(0.1f, 0.1f, 0.1f, 1.0f);
    static const ImVec4 CRLTextWhite(0.9f, 0.9f, 0.9f, 1.0f);
    static const ImVec4 CRL3DBackground(0.0f, 0.0f, 0.0f, 1.0f);


    struct GuiLayerUpdateInfo {
        bool firstFrame{};
        float width{};
        float height{};
        /**@brief Width of sidebar*/
        float sidebarWidth = 200.0f;
        /**@brief Size of elements in sidebar */
        float elementHeight = 140.0f;
        /**@brief Width of sidebar*/
        float debuggerWidth = 960.0f * 0.75f;
        /**@brief Size of elements in sidebar */
        float debuggerHeight = 480.0f * 0.75f;
        /**@brief Width of sidebar*/
        float metricsWidth = 150.0f;
        /**@brief Physical Graphics m_Device used*/
        std::string deviceName = "DeviceName";
        /**@brief Title of Application */
        std::string title = "TitleName";
        /**@brief array containing the last 50 entries of frametimes */
        std::array<float, 50> frameTimes{};
        /**@brief min/max values for frametimes, used for updating graph*/
        float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;
        /**@brief value for current frame timer*/
        float frameTimer{};
        uint64_t frameID = 0;
        /**@brief Font types used throughout the gui. usage: ImGui::PushFont(font13).. Initialized in GuiManager class */
        ImFont *font13{}, *font18{}, *font24{};

        std::vector<ImTextureID> imageButtonTextureDescriptor;

        // TODO crude and "quick" implementation. Lots of missed memory and uses way more memory than necessary. Fix in the future
        struct {
            unsigned char *pixels = nullptr;
            ImTextureID image[20]{};
            uint32_t id{};
            uint32_t lastFrame = 0;
            uint32_t width{};
            uint32_t height{};
            uint32_t imageSize{};
            uint32_t totalFrames{};
            uint32_t *delay{};
        } gif{};

        /** @brief ImGUI Overlay on previews window sizes */
        float viewAreaElementSizeY = {0};
        float viewAreaElementSizeX = {0};
        float previewBorderPadding = 60.0f;
        /** @brief add m_Device button params */
        float addDeviceBottomPadding = 90.0f;
        float addDeviceWidth = 180.0f, addDeviceHeight = 35.0f;
        /** @brief Height of popupModal*/
        float popupHeight = 600.0f;
        /** @brief Width of popupModal*/
        float popupWidth = 450.0f;
        /**@brief size of control area tabs*/
        float tabAreaHeight = 60.0f;
        /** @brief size of Control Area*/
        float controlAreaWidth = 440.0f, controlAreaHeight = height;
        int numControlTabs = 2;
        /** @brief size of viewing Area*/
        float viewingAreaWidth = width - controlAreaWidth - sidebarWidth, viewingAreaHeight = height;
        bool hoverState = false;
    };

    /** @brief Handle which is the communication from GUI to Scripts */
    struct GuiObjectHandles {
        /** @brief Handle for current devices located in sidebar */
        std::vector<Device> devices;
        /** @brief Static info used in creation of gui */
        std::unique_ptr<GuiLayerUpdateInfo> info{};

        /** User action to configure network automatically even when using manual approach **/
        bool configureNetwork = true;
        bool nextIsRemoteHead = false;
        /** Keypress and mouse events */
        float accumulatedActiveScroll = 0.0f;
        bool disableCameraRotationFromGUI = false;
        const Input *input{};
        std::array<float, 4> clearColor{};

        /*@brief Initialize \refitem clearColor because MSVC does not allow initializer list for std::array */
        GuiObjectHandles() {
            clearColor[0] = 0.870f;
            clearColor[1] = 0.878f;
            clearColor[2] = 0.862f;
            clearColor[3] = 1.0f;
        }

        bool showDebugWindow = false;
    };
}


#endif //MULTISENSE_DEFINITIONS_H
