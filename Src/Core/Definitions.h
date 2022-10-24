//
// Created by magnus on 15/05/21.
//

#ifndef MULTISENSE_DEFINITIONS_H
#define MULTISENSE_DEFINITIONS_H


#include "glm/glm.hpp"
#include "vulkan/vulkan_core.h"
#include "include/MultiSense/MultiSenseTypes.hh"

#include "MultiSense/Src/Tools/Logger.h"
#include "unordered_map"
#include "memory"
#include "GLFW/glfw3.h"
#include "Buffer.h"
#include "VulkanDevice.h"
#include "Camera.h"
#include "KeyInput.h"
#include <utility>

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
    AR_STATE_RESET = 4,
    AR_STATE_DISCONNECTED = 5,
    AR_STATE_UNAVAILABLE = 6,
    AR_STATE_DISCONNECT_AND_FORGET = 7,
    AR_STATE_REMOVE_FROM_LIST = 8,
    AR_STATE_JUST_ADDED = 9
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
        std::vector<ChannelInfo> channelInfo;
        CRLCameraBaseUnit baseUnit{};

        /**@brief location for which this device should save recorded frames **/
        std::string outputSaveFolder = "/Path/To/Folder/";
        bool isRecording = false;

        std::vector<std::string> attachedScripts{};
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

    /** @brief An initialized object needed to create a \refitem Device */
    struct EntryConnectDevice {
        std::string profileName = "VkRender";
        std::string IP = "10.66.171.21";
        std::string interfaceName;
        std::string description;
        uint32_t interfaceIndex{};

        std::string cameraName;

        EntryConnectDevice() = default;

        EntryConnectDevice(std::string ip, std::string iName, std::string camera, uint32_t idx, std::string desc) : IP(
                std::move(ip)),
                                                                                                                    interfaceName(
                                                                                                                            std::move(
                                                                                                                                    iName)),
                                                                                                                    description(
                                                                                                                            std::move(
                                                                                                                                    desc)),
                                                                                                                    interfaceIndex(
                                                                                                                            idx),
                                                                                                                    cameraName(
                                                                                                                            std::move(
                                                                                                                                    camera)) {
        }

        void reset() {
            profileName = "";
            IP = "";
            interfaceName = "";
            interfaceIndex = 0;
        }

        /**
         * @brief Utility function to check if the requested profile in \ref entry is not conflicting with any of the previously connected devices in the sidebar
         * @param devices list of current devices
         * @param entry new connection to be added
         * @return true of we can add this new profile to list. False if not
         */
        bool ready(const std::vector<VkRender::Device> &devices, const EntryConnectDevice &entry) const {
            bool profileNameEmpty = entry.profileName.empty();
            bool profileNameTaken = false;
            bool IPEmpty = entry.IP.empty();
            bool adapterNameEmpty = entry.interfaceName.empty();

            bool AdapterAndIPInTaken = false;



            // Loop through devices and check that it doesn't exist already.
            for (auto &d: devices) {
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
        explicit TextureData(CRLCameraDataType texType, uint32_t width, uint32_t height, bool manualMemoryMgmt = false) :
                m_Type(texType),
                m_Width(width),
                m_Height(height),
                m_Manual(manualMemoryMgmt) {

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

            if (m_Manual) {
                data = (uint8_t *) malloc(m_Len);
                data2 = (uint8_t *) malloc(m_Len2);
            }
        }

        ~TextureData() {
            if (m_Manual) {
                free(data);
                free(data2);
            }
        }
        CRLCameraDataType m_Type = AR_CAMERA_IMAGE_NONE;
        uint32_t m_Width= 0;
        uint32_t m_Height = 0;
        uint8_t *data{};
        uint8_t *data2{};
        uint32_t m_Len = 0, m_Len2 = 0 , m_Id{}, m_Id2{};

    private:
        bool m_Manual = true;

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
        Buffer bufferOne;
        Buffer bufferTwo;
        Buffer bufferThree;
        Buffer bufferFour;
    };

    /** containing Basic Vulkan Resources for rendering **/
    struct RenderUtils {
        VulkanDevice *device{};
        uint32_t UBCount = 0;
        VkRenderPass *renderPass{};
        std::vector<UniformBufferSet> uniformBuffers;
        const VkRender::ObjectPicking *picking;
    };

    /**@brief grouping containing useful pointers used to render scripts. This will probably change frequently as the viewer grows **/
    struct RenderData {
        uint32_t index;
        Camera *camera = nullptr;
        float deltaT = 0.0f;
        bool drawThisScript = false;
        float scriptRuntime = 0.0f;
        int scriptDrawCount = 0;
        std::string scriptName;
        std::unique_ptr<MultiSense::CRLPhysicalCamera> *crlCamera;
        ScriptType type;
        Log::Logger *pLogger;
        uint32_t height;
        uint32_t width;
        const Input *input;
    };


}


#endif //MULTISENSE_DEFINITIONS_H
