//
// Created by magnus on 10/3/23.
//

#ifndef MULTISENSE_VIEWER_DATACAPTURE_H
#define MULTISENSE_VIEWER_DATACAPTURE_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"


class DataCapture: public VkRender::Base, public VkRender::RegisteredInFactory<DataCapture>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    DataCapture() {
        DISABLE_WARNING_PUSH
                DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
                s_bRegistered;
        DISABLE_WARNING_POP
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of Grid **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<DataCapture>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "DataCapture"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    void onUIUpdate(VkRender::GuiObjectHandles *uiHandle) override;
    /** @brief destroy function called before script deletion **/
    void onDestroy() override{
    }

    struct ImageData {
        int imageID;
        double qw, qx, qy, qz; // Quaternion components
        double tx, ty, tz;     // Translation vector
        int cameraID;
        std::string imageName;
        bool savedToFile = false;
    };
    struct CameraData {
        int cameraID;
        std::string model;
        int width;
        int height;
        std::vector<double> parameters; // To store variable-length camera parameters.
    };

    struct Scene {
        std::vector<ImageData> poses;
        std::filesystem::path objPath;
        std::string scene;
        CameraData camera;
        bool loaded = false;
        bool posesReady = false;
        bool exists = false;
    };
    bool recordNextScene = false;
    uint32_t sceneIndex = 0;
    uint32_t poseIndex = 0;
    uint32_t frameCount = 1;
    uint32_t skipFrames = 5;
    std::vector<Scene> scenes;
    std::vector<ImageData> images;
    std::vector<std::string> entities;
    bool resetDataCapture = true;
    bool saveDepthImage = false;

    std::vector<ImageData> loadPoses(std::filesystem::path path);

    static bool compareImageID(const ImageData &img1, const ImageData &img2);

    std::vector<CameraData> loadCameraParams(const std::filesystem::path &path);

    void loadColmapPoses(VkRender::GuiObjectHandles *uiHandle);

    void setCOLMAPPose();

    void deleteDataCaptureScenes();

    static bool compareSceneNumber(const Scene &s1, const Scene &s2);
};

#endif //MULTISENSE_VIEWER_DATACAPTURE_H
