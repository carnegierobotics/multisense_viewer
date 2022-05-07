//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_LAYER_H
#define MULTISENSE_LAYER_H

#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>
#include "imgui_internal.h"
#include "imgui.h"


struct GuiLayerUpdateInfo {
    bool firstFrame{};
    float frameTimer{};
    float width{};
    float height{};
    float sidebarWidth = 250.0f;
    std::string deviceName = "DeviceName";
    std::string title = "TitleName";

    std::array<float, 50> frameTimes{};
    float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;

    ImFont *font13{}, *font18{}, *font24{};

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
    std::string name;
    std::string cameraName;
    std::string IP;

    /** @brief Connection state and selection */
    bool clicked;
    ArConnectionState state;

    /** @brief Camera streaming modes  */
    std::vector<StreamingModes> modes;
    /** @brief Which mode is currently selected */
    std::string selectedStreamingMode = "Select sensor resolution";

    /** @brief  */
    bool pointCloud = false;
    bool depthImage = false;
    bool colorImage = false;

};

/** @brief Handle which is the communication from GUI to Scripts */
struct GuiObjectHandles {
    std::vector<Element> *devices{};
    GuiLayerUpdateInfo *info{};
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
