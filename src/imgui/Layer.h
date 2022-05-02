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

struct Element {
    std::string name;
    std::string cameraName;
    std::string IP;

    bool clicked;
    ArConnectionState state;

    CRLPhysicalCamera *camera;

};

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
