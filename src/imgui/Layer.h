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

    /** @brief Height of popupModal*/
    float popupHeight = 350.0f;
    /** @brief Width of popupModal*/
    float popupWidth = 500.0f;
};

typedef enum {
    ArConnectedState,
    ArConnectingState,
    ArActiveState,
    ArInActiveState,
    ArDisconnectedState,
    ArUnavailableState
} ArConnectionState;

/** @brief  */
struct StreamingModes {
    std::string label;
    std::string modeName;

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
    /** @brief Camera streaming modes  */
    std::vector<StreamingModes> modes;
    /** @brief Which mode is currently selected */
    std::string selectedStreamingMode = "Select sensor resolution";

    /** @brief  Showing point cloud view*/
    bool pointCloud = false;
    /** @brief  Showing depth image stream*/
    bool depthImage = false;
    /** @brief  Showing color image stream*/
    bool colorImage = false;

    /** @brief Show a default preview with some selected streams*/
    bool btnShowPreviewBar = false;
};

/** @brief Handle which is the communication from GUI to Scripts */
struct GuiObjectHandles {
    /** @brief Handle for current devices located in sidebar */
    std::vector<Element> *devices{};
    /** @brief Static info used in creation of gui */
    GuiLayerUpdateInfo *info{};


    float sliderOne = 0.0f;
    float sliderTwo = 0.0f;
    float sliderThree = 0.0f;

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
