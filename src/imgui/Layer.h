//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_LAYER_H
#define MULTISENSE_LAYER_H

#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>
#include "imgui_internal.h"
#include "imgui.h"
#include <array>


struct GuiLayerUpdateInfo {
    bool firstFrame{};
    float width{};
    float height{};
    /**@brief Width of sidebar*/
    float sidebarWidth = 250.0f;
    /**@brief Size of elements in sidebar */
    float elementHeight = 140.0f;
    /**@brief Physical Graphics device used*/
    std::string deviceName = "DeviceName";
    /**@brief Title of Application */
    std::string title = "TitleName";
    /**@brief array containing the last 50 entries of frametimes */
    std::array<float, 50> frameTimes{};
    /**@brief min/max values for frametimes, used for updating graph*/
    float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;
    /**@brief value for current frame timer*/
    float frameTimer{};
    /**@brief Font types used throughout the gui. usage: ImGui::PushFont(font13).. Initialized in GuiManager class */
    ImFont *font13{}, *font18{}, *font24{};

    ImTextureID imageButtonTextureDescriptor[9];
    float offset5px = 5.0f;

    /** @brief Height of popupModal*/
    float popupHeight = 350.0f;
    /** @brief Width of popupModal*/
    float popupWidth = 500.0f;

    /** @brief size of Control Area*/
    float controlAreaWidth = 440.0f, controlAreaHeight = height;

};

typedef enum {
    ArConnectedState,
    ArConnectingState,
    ArActiveState,
    ArInActiveState,
    ArDisconnectedState,
    ArUnavailableState,
    ArJustAddedState
} ArConnectionState;

typedef enum StreamIndex {
    // First 0 - 3 elements also correspond to array indices. Should not change.
    PREVIEW_LEFT = 0,
    PREVIEW_RIGHT = 1,
    PREVIEW_DISPARITY = 2,
    PREVIEW_AUXILIARY = 3,
    PREVIEW_VIRTUAL = 4 ,

    // Other flags
    PREVIEW_PLAYING = 10,
    PREVIEW_PAUSED = 11,
    PREVIEW_STOPPED = 12,
    PREVIEW_NONE = 13,
    PREVIEW_RESET = 14,
} CameraStreamInfoFlag;

typedef enum page {
    PAGE_PREVIEW_DEVICES = 0,
    PAGE_DEVICE_INFORMATION = 1,
    PAGE_CONFIGURE_DEVICE = 2,
    PAGE_TOTAL_PAGES = 3,
    TAB_NONE = 10,
    TAB_2D_PREVIEW = 11,
    TAB_3D_POINTCLOUD = 12,
} Page;

/** @brief  */
struct StreamingModes {
    /** @brief Which gui index is selected */
    CameraStreamInfoFlag streamIndex = PREVIEW_LEFT;
    /** @brief Current camera streaming state  */
    CameraStreamInfoFlag playbackStatus = PREVIEW_NONE;

    /** @brief Camera streaming modes  */
    std::vector<std::string> modes;
    /** @brief Camera streaming sources  */
    std::vector<std::string> sources;
    uint32_t selectedModeIndex = 0;
    uint32_t selectedSourceIndex = 0;
    /** @brief Which mode is currently selected */
    std::string selectedStreamingMode = "Select sensor resolution";
    /** @brief Which source is currently selected */
    std::string selectedStreamingSource = "Select sensor resolution";


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
    /** @brief Flag for registering if device is clicked in sidebar */
    bool clicked;
    /** @brief Current connection state for this device */
    ArConnectionState state;

    std::unordered_map<int, StreamingModes> streams;


    Page selectedPreviewTab = TAB_NONE;
    /** @brief  Showing point cloud view*/
    bool pointCloud = false;
    /** @brief  Showing depth image stream*/
    bool depthImage = false;
    /** @brief  Showing color image stream*/
    bool colorImage = false;

    /** @brief Show a default preview with some selected streams*/
    bool button = false;
};

/** @brief Handle which is the communication from GUI to Scripts */
struct GuiObjectHandles {
    /** @brief Handle for current devices located in sidebar */
    std::vector<Element> *devices = new std::vector<Element>();
    /** @brief Static info used in creation of gui */
    GuiLayerUpdateInfo *info{};

    float sliderOne = -1.77f;
    float sliderTwo = 0.986f;
    float sliderThree = -0.718;

};

class Layer {

public:

    virtual ~Layer() = default;

    virtual void OnAttach() {}

    virtual void OnDetach() {}

    virtual void OnUIRender(GuiObjectHandles *handles) {}

    virtual void onFinishedRender() {}

};


#endif //MULTISENSE_LAYER_H
