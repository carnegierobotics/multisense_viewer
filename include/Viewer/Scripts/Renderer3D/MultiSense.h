//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_SCRIPT_MULTISENSE_CAMERA
#define MULTISENSE_VIEWER_SCRIPT_MULTISENSE_CAMERA

#include "Viewer/Scripts/Private/ScriptBuilder.h"



class MultiSense: public VkRender::Base, public VkRender::RegisteredInFactory<MultiSense>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    MultiSense() {
        DISABLE_WARNING_PUSH
                DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
                s_bRegistered;
        DISABLE_WARNING_POP
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of MultiSense **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<MultiSense>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "MultiSense"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief destroy function called before script deletion **/
    void onDestroy() override{
    }

    struct LightSource {
        glm::vec3 color = glm::vec3(1.0f);
        glm::vec3 rotation = glm::vec3(75.0f, 40.0f, 0.0f);
    } lightSource;

};


#endif //MULTISENSE_VIEWER_SCRIPT_MULTISENSE_CAMERA
