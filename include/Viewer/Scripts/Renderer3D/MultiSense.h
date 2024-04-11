//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_SCRIPT_MULTISENSE_CAMERA
#define MULTISENSE_VIEWER_SCRIPT_MULTISENSE_CAMERA

#include "Viewer/Scripts/Private/ScriptBuilder.h"

class MultiSense: public VkRender::Base2, public VkRender::RegisteredInFactory<MultiSense>
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
    static std::unique_ptr<Base2> CreateMethod() { return std::make_unique<MultiSense>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "MultiSense"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief destroy function called before script deletion **/
    void onDestroy() override{
    }

    /** @brief draw function called once per frame **/
    void draw(CommandBuffer * commandBuffer, uint32_t i, bool b) override;




};


#endif //MULTISENSE_VIEWER_SCRIPT_MULTISENSE_CAMERA
