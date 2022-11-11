//
// Created by magnus on 10/12/22.
//

#ifndef MULTISENSE_VIEWER_SLAM_H
#define MULTISENSE_VIEWER_SLAM_H



#include <MultiSense/Src/Scripts/Private/ScriptBuilder.h>
#include "MultiSense/Src/Renderer/Renderer.h"
#include "MultiSense/Src/VO/Features/VisualOdometry.h"
#include "MultiSense/Src/ModelLoaders/glTFModel.h"
#include "MultiSense/Src/VO/GraphSlam.h"

class SLAM: public VkRender::Base, public VkRender::RegisteredInFactory<SLAM>, glTFModel
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    SLAM() {
        s_bRegistered;
    }
    ~SLAM() = default;
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<SLAM>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "SLAM"; }

    /** @brief Setup function called one during engine prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief Get the type of script. This will determine how it interacts with a gameobject **/
    ScriptType getType() override {return type;}

    void onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) override;

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptType type = AR_SCRIPT_TYPE_RENDER;
    void draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) override;
    std::unique_ptr<glTFModel::Model> m_Model;

    std::vector<std::unique_ptr<glTFModel::Model>> m_TruthTraces;

    std::vector<std::string> leftFileNames;
    std::vector<std::string> rightFileNames;
    std::vector<std::string> depthFileNames;

    std::map<size_t, GSlam::FeatureSet> m_FeatureLeftMap{}, m_FeatureRightMap{};
    std::map<size_t, cv::Mat> m_LMap{};
    std::map<size_t, cv::Mat> m_RMap{};
    std::map<size_t, cv::Mat> m_DMap{};

    cv::Mat m_PLeft, m_PRight;

    size_t id = 0;
    size_t frame = 150;
    cv::Mat m_Rotation;
    glm::mat4 m_RotationMat;
    glm::mat4 m_TranslationMat;
    cv::Mat m_Translation;
    cv::Mat m_Pose;
    cv::Mat m_Trajectory;

    struct gtPos{
        float x{}, y{}, z{};

        [[nodiscard]] glm::vec3 getVec() const{
            return {x, y, z};
        }
    };

    VkRender::Shared shared;
    std::vector<gtPos> gtPositions{};

    void fromCV2GLM(const cv::Mat &cvmat, glm::mat4 *glmmat);
};


#endif //MULTISENSE_VIEWER_SLAM_H
