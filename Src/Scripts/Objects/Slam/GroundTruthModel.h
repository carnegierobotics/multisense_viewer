//
// Created by magnus on 11/10/22.
//

#ifndef MULTISENSE_VIEWER_GROUNDTRUTHMODEL_H
#define MULTISENSE_VIEWER_GROUNDTRUTHMODEL_H


#include <MultiSense/Src/Scripts/Private/ScriptBuilder.h>
#include "MultiSense/Src/Renderer/Renderer.h"
#include "MultiSense/Src/VO/Features/VisualOdometry.h"
#include "MultiSense/Src/ModelLoaders/glTFModel.h"
#include "MultiSense/Src/VO/GraphSlam.h"

class GroundTruthModel: public VkRender::Base, public VkRender::RegisteredInFactory<GroundTruthModel>, glTFModel
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    GroundTruthModel() {
        s_bRegistered;
    }
    ~GroundTruthModel() = default;
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<GroundTruthModel>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "GroundTruthModel"; }

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

    std::unique_ptr<glTFModel::Model> m_TruthModel;
    struct gtPos{
        float x{}, y{}, z{};
        double time;

        struct{
            float x{}, y{}, z{}, w{};
        }orientation;

        glm::vec3 getVec() const{
            return {x, y, z};
        }
    };

    VkRender::Shared* shared;

    std::vector<gtPos> gtPositions{};

    std::vector<std::unique_ptr<glTFModel::Model>> m_Traces;


    double findClosest(const std::vector<gtPos>& arr, size_t n, double target);

    double getClosest(double val1, double val2, double target);
};




#endif //MULTISENSE_VIEWER_GROUNDTRUTHMODEL_H
