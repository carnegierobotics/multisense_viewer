//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_LAYER_H
#define MULTISENSE_LAYER_H

#define IMGUI_INCLUDE_IMGUI_USER_H


#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>
#include "imgui.h"
#include <array>
#include "map"

namespace AR {

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

        float viewAreaElementPositionsY[9] = {0};
        float viewAreaElementSizeY = {0};
        float viewAreaElementSizeX = {0};
        float previewBorderPadding = 60.0f;

        std::unordered_map<ImageElementIndex, ArEngine::ImageElement> img;

        std::vector<ArEngine::ImageElement> imageElements;

        /** @brief add device button params */
        float addDeviceBottomPadding = 70.0f, addDeviceLeftPadding = 20.0f;
        float addDeviceWidth = 200.0f, addDeviceHeight = 35.0f;

        /** @brief Height of popupModal*/
        float popupHeight = 400.0f;
        /** @brief Width of popupModal*/
        float popupWidth = 450.0f;

        /**@brief size of control area tabs*/
        float tabAreaHeight = 60.0f;

        /** @brief size of Control Area*/
        float controlAreaWidth = 440.0f, controlAreaHeight = height;
        int numControlTabs = 4;

        /** @brief size of viewing Area*/
        float viewingAreaWidth = 590.0f, viewingAreaHeight = height;

        /** @brief control area drop down box size */
        float controlDropDownWidth = 350.0, controlDropDownHeight = 40.0f;
        float dropDownWidth = 300.0f, dropDownHeightOpen = 220.0f;

    };


/** @brief  */
    struct StreamingModes {
        /** @brief Name of this streaming mode (i.e: front camera) */
        std::string name;
        std::string attachedScript;
        /** @brief Which gui index is selected */
        CameraStreamInfoFlag streamIndex = AR_PREVIEW_LEFT;
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
        CRLCameraResolution selectedStreamingMode;
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

/** @brief Handle which is the communication from GUI to Scripts */
    struct GuiObjectHandles {
        /** @brief Handle for current devices located in sidebar */
        std::vector<Element> *devices = new std::vector<Element>();
        /** @brief Static info used in creation of gui */
        GuiLayerUpdateInfo *info{};

        /** Keypress and mouse events */
        int keypress; // TODO Redo usage of keypress. Was only meant for quick debugging
        ArEngine::MouseButtons mouseBtns;
        float accumulatedMouseScroll = 0.0f;
        float mouseScrollSpeed = 50.0f;

    };

    static ImVec4 yellow(0.98f, 0.65f, 0.00f, 1.0f);
    static ImVec4 green(0.26f, 0.42f, 0.31f, 1.0f);
    static ImVec4 TextGreenColor(0.16f, 0.95f, 0.11f, 1.0f);
    static ImVec4 TextRedColor(0.95f, 0.045f, 0.041f, 1.0f);

    static ImVec4 red(0.613f, 0.045f, 0.046f, 1.0f);
    static ImVec4 DarkGray(0.1f, 0.1f, 0.1f, 1.0f);
    static ImVec4 PopupBackground(0.084f, 0.231f, 0.389f, 1.0f);
    static ImVec4 PopupTextInputBackground(0.01f, 0.05f, 0.1f, 1.0f);
    static ImVec4 TextColorGray(0.75f, 0.75f, 0.75f, 1.0f);

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
