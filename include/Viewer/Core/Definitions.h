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


#include <unordered_map>
#include <memory>
#include <utility>
#include <array>
#include <vulkan/vulkan_core.h>
#include <MultiSense/MultiSenseTypes.hh>
#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>
#include <imgui/imgui.h>

#include "Viewer/Core/KeyInput.h"
#include "Viewer/Core/Buffer.h"
#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Camera.h"
#include "Viewer/Core/Texture.h"

#define INTERVAL_10_SECONDS 10
#define INTERVAL_1_SECOND 1
#define MAX_IMAGES_IN_QUEUE 30


// Predeclare to speed up compile times
namespace VkRender::MultiSense {
    class CRLPhysicalCamera;
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
    /**
     * Create this script before default and always render this type first. No internal ordering amongst scripts
     */
    CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE,
} ScriptType;

/**
 * @brief Labels data coming from the camera to a type used to initialize textures with various formats and samplers
 */
typedef enum CRLCameraDataType {
    CRL_POINT_CLOUD,
    CRL_GRAYSCALE_IMAGE,
    CRL_COLOR_IMAGE,
    CRL_COLOR_IMAGE_YUV420,
    CRL_YUV_PLANAR_FRAME,
    CRL_CAMERA_IMAGE_NONE,
    CRL_DISPARITY_IMAGE
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
    CRL_STATE_JUST_ADDED = 11               /// Skip to connection flow, just added into sidebar list
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
    FLOAT_SLIDER = 0,
    INT_SLIDER = 1,
    INPUT_NUMBER = 2,
} ScriptWidgetType;

/**
 * @brief Adjustable sensor parameters
 */
struct WhiteBalanceParams {
    float whiteBalanceRed = 1.0f;
    float whiteBalanceBlue = 1.0f;
    bool autoWhiteBalance = true;
    uint32_t autoWhiteBalanceDecay = 3;
    float autoWhiteBalanceThresh = 0.5f;
    bool update = false;
};

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
    crl::multisense::DataSource exposureSource = crl::multisense::Source_Luma_Left;

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
    ExposureParams ep{};
    WhiteBalanceParams wb{};
    LightingParams light{};
    CalibrationParams calib{};

    float gain = 1.0f;
    float fps = 30.0f;
    float stereoPostFilterStrength = 0.5f;
    bool hdrEnabled = false;
    float gamma = 2.0f;

    bool update = false;
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
        uint32_t x{}, y{};
        uint32_t r{}, g{}, b{};
        uint32_t intensity{};
        float depth{};
    };

    /**
     * @brief Information block for each Preview window, should support 1-9 possible previews. Contains stream names and info to presented to the user.
     * Contains a mix of GUI related and MultiSense Device related info.
     */
    struct PreviewWindow {
        std::vector<std::string> availableSources{}; // Human-readable names of camera sources
        std::vector<std::string> availableRemoteHeads{};
        std::string selectedSource = "Source";
        uint32_t selectedSourceIndex = 0;
        crl::multisense::RemoteHeadChannel selectedRemoteHeadIndex = 0;
        float xPixelStartPos = 0;
        float yPixelStartPos = 0;
        float row = 0;
        float col = 0;
    };

    /**
     * @brief Ties together the preview windows for each MultiSense channel
     * Contain possible streams and resolutions for each channel
     */
    struct ChannelInfo {
        uint32_t index = 0;
        std::vector<std::string> availableSources{};
        ArConnectionState state = CRL_STATE_DISCONNECTED;
        std::vector<std::string> modes{};
        uint32_t selectedModeIndex = 0;
        CRLCameraResolution selectedMode = CRL_RESOLUTION_NONE;
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
        /** @brief Current connection state for this device */
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
        /**@brief Flag to decide if user is currently recording frames */
        bool isRecording = false;
        /** @brief 3D view camera type for this device. Arcball or first person view controls) */
        int cameraType = 0;
        /** @brief Reset 3D view camera position and rotation */
        bool resetCamera = false;
        /** @brief Pixel information, on mouse hover for textures */
        CursorPixelInformation pixelInfo{};
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
        bool useIMU = false;

        /** @brief Following is UI elements settings for the active device **/
        /** @brief Which TAB this preview has selected. 2D or 3D view. */
        Page selectedPreviewTab = CRL_TAB_2D_PREVIEW;
        /** @brief What type of layout is selected for this device*/
        PreviewLayout layout = CRL_PREVIEW_LAYOUT_SINGLE;
        /** @brief IF 3D area should be extended or not */
        bool extend3DArea = false;
        Device() {
            outputSaveFolder.resize(255);
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
        float wheel = 0.0f; // to initialize arcball zoom
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
        float exposure = 4.5f;
        float gamma = 2.2f;
        float prefilteredCubeMipLevels = 0.0f;
        float scaleIBLAmbient = 1.0f;
        float debugViewInputs = 0.0f;
        float debugViewEquation = 0.0f;
        glm::vec4 lightColor{};
        glm::vec4 objectColor{};
        glm::vec4 lightPos{};
        glm::vec4 viewPos{};
    };

    struct SkyboxTextures {
        TextureCubeMap environmentMap{};
        TextureCubeMap irradianceCube{};
        TextureCubeMap prefilterEnv{};
        Texture2D lutBrdf{};
        int prefilteredCubeMipLevels = 0;
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
            int prefilteredCubeMipLevels = 0;
        } skybox;
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
        const MultiSense::CRLPhysicalCamera *crlCamera{};
        ScriptType type{};
        uint32_t height = 0;
        uint32_t width = 0;
        const Input *input = nullptr;
        bool additionalBuffers = false;
    };

    /** @brief Set of Default colors */
    namespace Colors {
        static const ImVec4 yellow(0.98f, 0.65f, 0.00f, 1.0f);
        static const ImVec4 green(0.26f, 0.42f, 0.31f, 1.0f);
        static const ImVec4 TextGreenColor(0.16f, 0.95f, 0.11f, 1.0f);
        static const ImVec4 TextRedColor(0.95f, 0.345f, 0.341f, 1.0f);
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
        static const ImVec4 CRLTextLightGray(0.4f, 0.4f, 0.4f, 1.0f);
        static const ImVec4 CRLTextWhite(0.9f, 0.9f, 0.9f, 1.0f);
        static const ImVec4 CRLTextWhiteDisabled(0.75f, 0.75f, 0.75f, 1.0f);
        static const ImVec4 CRL3DBackground(0.0f, 0.0f, 0.0f, 1.0f);
    }

    struct GuiLayerUpdateInfo {
        bool firstFrame{};
        /** @brief Width of window surface */
        float width{};
        /** @brief Height of window surface */
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
        float metricsWidth = 350.0f;
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
        /** @brief Current frame*/
        uint64_t frameID = 0;
        /**@brief Font types used throughout the gui. usage: ImGui::PushFont(font13).. Initialized in GuiManager class */
        ImFont *font13{}, *font15, *font18{}, *font24{};

        /** @brief Containing descriptor handles for each image button texture */
        std::vector<ImTextureID> imageButtonTextureDescriptor;

        /** @brief
        * Container to hold animated gif images
        */
        struct {
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
        float popupWidth = 550.0f;
        /**@brief size of control area tabs*/
        float tabAreaHeight = 60.0f;
        /** @brief size of Control Area*/
        float controlAreaWidth = 440.0f, controlAreaHeight = height;
        int numControlTabs = 2;
        /** @brief size of viewing Area*/
        float viewingAreaWidth = width - controlAreaWidth - sidebarWidth, viewingAreaHeight = height;
        bool hoverState = false;
    };

    /** @brief Handle which is the MAIN link between ''frontend and backend'' */
    struct GuiObjectHandles {
        /** @brief Handle for current devices located in sidebar */
        std::vector<Device> devices;
        /** @brief GUI window info used for creation and updating */
        std::unique_ptr<GuiLayerUpdateInfo> info{};
        /** User action to configure network automatically even when using manual approach **/
        bool configureNetwork = false;
        /** Keypress and mouse events */
        float accumulatedActiveScroll = 0.0f;
        /** @brief min/max scroll used in DoublePreview layout */
        float maxScroll = 500.0f, minScroll = -850.0f;
        /** @brief when a GUI window is hovered dont move the camera in the 3D view */
        bool disableCameraRotationFromGUI = false;
        /** @brief Input from backend to IMGUI */
        const Input *input{};
        std::array<float, 4> clearColor{};
        /** @brief Display the debug window */
        bool showDebugWindow = false;

        /** @brief Initialize \refitem clearColor because MSVC does not allow initializer list for std::array */
        GuiObjectHandles() {
            clearColor[0] = 0.870f;
            clearColor[1] = 0.878f;
            clearColor[2] = 0.862f;
            clearColor[3] = 1.0f;
        }

    };

}


#endif //MULTISENSE_DEFINITIONS_H
