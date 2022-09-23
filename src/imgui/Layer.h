//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_LAYER_H
#define MULTISENSE_LAYER_H

#define IMGUI_INCLUDE_IMGUI_USER_H
#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS

#include "imgui.h"
#include <array>
#include "map"
#include "unordered_map"
#include "string"
#include "MultiSense/src/core/Definitions.h"
#include "MultiSense/src/core/KeyInput.h"

namespace AR {

    struct GuiLayerUpdateInfo {
        bool firstFrame{};
        float width{};
        float height{};
        /**@brief Width of sidebar*/
        float sidebarWidth = 200.0f;
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

        ImTextureID imageButtonTextureDescriptor[10];

        // TODO crude and "quick" implementation. Lots of missed memory and uses way more memory than necessary. Fix in the future
        struct {
            unsigned char* pixels = nullptr;
            ImTextureID image[20]{};
            uint32_t id{};
            uint32_t lastFrame = 0;
            uint32_t width{};
            uint32_t height{};
            uint32_t imageSize{};
            uint32_t totalFrames{};
            uint32_t* delay{};
        } gif{};

        float viewAreaElementPositionsY[9] = {0};
        float viewAreaElementSizeY = {0};
        float viewAreaElementSizeX = {0};

        float previewBorderPadding = 60.0f;

        /** @brief add device button params */
        float addDeviceBottomPadding = 70.0f, addDeviceLeftPadding = 20.0f;
        float addDeviceWidth = 200.0f, addDeviceHeight = 35.0f;

        /** @brief Height of popupModal*/
        float popupHeight = 500.0f;
        /** @brief Width of popupModal*/
        float popupWidth = 450.0f;

        /**@brief size of control area tabs*/
        float tabAreaHeight = 60.0f;

        /** @brief size of Control Area*/
        float controlAreaWidth = 440.0f, controlAreaHeight = height;
        int numControlTabs = 4;

        /** @brief size of viewing Area*/
        float viewingAreaWidth = width - controlAreaWidth - sidebarWidth, viewingAreaHeight = height;

        /** @brief control area drop down box size */
        float controlDropDownWidth = 350.0, controlDropDownHeight = 40.0f;
        float dropDownWidth = 300.0f, dropDownHeightOpen = 220.0f;

        bool hoverState = false;
    };



/** @brief Handle which is the communication from GUI to Scripts */
    struct GuiObjectHandles {
        /** @brief Handle for current devices located in sidebar */
        std::vector<Element> *devices = new std::vector<Element>();
        /** @brief Static info used in creation of gui */
        GuiLayerUpdateInfo *info{};

        /** Keypress and mouse events */
        const ArEngine::MouseButtons* mouseBtns{};
        float accumulatedActiveScroll = 0.0f;
        const Input* input{};

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
    static const ImVec4 CRLGray424Main(0.462f, 0.474f, 0.494f, 1.0f);
    static const ImVec4 CRLDarkGray425(0.301f, 0.313f, 0.309f, 1.0f);
    static const ImVec4 CRLRed(0.768f, 0.125f, 0.203f, 1.0f);
    static const ImVec4 CRLRedHover(0.86f, 0.378f, 0.407f, 1.0f);
    static const ImVec4 CRLRedActive(0.96f, 0.478f, 0.537f, 1.0f);

    static const ImVec4 CRLBlueIsh(0.313f, 0.415f, 0.474f, 1.0f);
    static const ImVec4 CRLTextGray(0.1f, 0.1f, 0.1f, 1.0f);


    class Layer {

    public:

        virtual ~Layer() = default;

        virtual void OnAttach() {}

        virtual void OnDetach() {}

        virtual void OnUIRender(GuiObjectHandles *handles) {}

        virtual void onFinishedRender() {}

    };
};

#endif //MULTISENSE_LAYER_H
