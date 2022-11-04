//
// Created by magnus on 11/3/22.
//

#ifndef MULTISENSE_VIEWER_MAP_H
#define MULTISENSE_VIEWER_MAP_H




#include <MultiSense/Src/Scripts/Private/ScriptBuilder.h>
#include "MultiSense/Src/Renderer/Renderer.h"
#include "MultiSense/Src/VO/Features/VisualOdometry.h"
#include "MultiSense/Src/ModelLoaders/glTFModel.h"
#include "MultiSense/Src/VO/GraphSlam.h"

class Map: public VkRender::Base, public VkRender::RegisteredInFactory<Map>, glTFModel
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    Map() {
        s_bRegistered;
    }
    ~Map() = default;
    /** @brief Static method to create class, returns a unique ptr of Terrain **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<Map>(); }
    /** @brief Name which is registered for this class. Same as ClassName **/
    static std::string GetFactoryName() { return "Map"; }

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

    std::vector<std::unique_ptr<glTFModel::Model>> m_TruthTraces;


    cv::Mat m_PLeft, m_PRight;
    size_t frame = 0;
    size_t drawBoxes = 0;
    struct gtPos{
        float x{}, y{}, z{};

        glm::vec3 getVec() const{
            return {x, y, z};
        }
    };
    std::vector<gtPos> gtPositions{};

    void recv(void* data) override;
};




#endif //MULTISENSE_VIEWER_MAP_H
