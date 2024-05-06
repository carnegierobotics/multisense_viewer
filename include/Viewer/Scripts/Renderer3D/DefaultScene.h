//
// Created by magnus on 10/3/23.
//

#ifndef MULTISENSE_VIEWER_DEFAULTSCENE_H
#define MULTISENSE_VIEWER_DEFAULTSCENE_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"


class DefaultScene: public VkRender::Base, public VkRender::RegisteredInFactory<DefaultScene>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    DefaultScene() {
        DISABLE_WARNING_PUSH
                DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
                s_bRegistered;
        DISABLE_WARNING_POP
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of Grid **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<DefaultScene>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "DefaultScene"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    void onUIUpdate(VkRender::GuiObjectHandles *uiHandle) override;

    void onWindowResize(const VkRender::GuiObjectHandles *uiHandle) override;
    /** @brief destroy function called before script deletion **/
    void onDestroy() override{
    }
    /** @brief draw function called once per frame **/
    void draw(CommandBuffer * commandBufobjectfer, uint32_t i, bool b) override;

    bool enable = true;
    bool hide = false;
    bool showDepthView = true;

};

#endif //MULTISENSE_VIEWER_DEFAULTSCENE_H
