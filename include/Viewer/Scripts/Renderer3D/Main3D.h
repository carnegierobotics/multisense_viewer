//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_MAIN3D_H
#define MULTISENSE_VIEWER_MAIN3D_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"
#include "Viewer/ModelLoaders/CustomModels.h"


class Main3D: public VkRender::Base, public VkRender::RegisteredInFactory<Main3D>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    Main3D() {
        DISABLE_WARNING_PUSH
                DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
                s_bRegistered;
        DISABLE_WARNING_POP
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of Main3D **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<Main3D>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "Main3D"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief destroy function called before script deletion **/
    void onDestroy() override{
        humvee.reset();
    }
    /** @brief set if this script should be drawn or not. */
    void setDrawMethod(DrawMethod _drawMethod) override{ this->drawMethod = _drawMethod; }

    /** @brief draw function called once per frame **/
    void draw(CommandBuffer * commandBuffer, uint32_t i, bool b) override;

    /** @brief Get the type of script. Future extension if Scripts should behave differently **/
    ScriptTypeFlags getType() override { return type; }
    DrawMethod getDrawMethod() override {return drawMethod;}

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    ScriptTypeFlags type = CRL_SCRIPT_TYPE_RENDERER3D;
    DrawMethod drawMethod = CRL_SCRIPT_DONT_DRAW;

    std::unique_ptr<GLTFModel::Model> humvee;


    struct LightSource {
        glm::vec3 color = glm::vec3(1.0f);
        glm::vec3 rotation = glm::vec3(75.0f, 40.0f, 0.0f);
    } lightSource;

    char buf[1024] = "/home/magnus/crl/disparity_quality/processing/images_and_pose.csv";
    char filePathDialog[1024] = "/home/magnus/crl/disparity_quality/processing/images_and_pose.csv";
    bool openDialog = false;

    bool play = false;
    bool pause = false;
    bool paused = false;
    bool restart = false;
    std::string simTimeText = "1";
    int val = 1;

    struct Data {
        std::chrono::system_clock::time_point timePoint;
        std::chrono::duration<double> timeDelta;
        std::chrono::duration<double> dt;
        std::string timestamp;
        float x, y, z;
        float qw, qx, qy, qz;
    };

    std::vector<Data> entries;
    std::chrono::steady_clock::time_point lastEntryTime;
    std::chrono::steady_clock::time_point startPlay;
    size_t entryIdx = 0;

    std::chrono::system_clock::time_point convertToTimePoint(const std::string& timestamp) {
        std::tm tm = {};
        int nanoseconds;

        // Parse the main timestamp without milliseconds
        std::istringstream ss(timestamp);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

        // Skip the dot and parse milliseconds
        ss.ignore();
        ss >> nanoseconds;

        // Convert to time_point and add the nanoseconds
        std::chrono::system_clock::time_point timePointWithoutMs = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        auto nanoDuration = std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::nanoseconds(nanoseconds));
        std::chrono::system_clock::time_point pt = timePointWithoutMs + nanoDuration;
        return pt;
    }


    bool forceRealTime = true;
};


#endif //MULTISENSE_VIEWER_MAIN3D_H
