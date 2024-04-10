//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_MULTISENSE_H
#define MULTISENSE_VIEWER_MULTISENSE_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/ModelLoaders/GLTFModel.h"
#include "Viewer/ModelLoaders/CustomModels.h"
#include "Viewer/ModelLoaders/ImageView.h"
#include "Viewer/SYCL/SyclRenderer.h"

class ImageViewer: public VkRender::Base, public VkRender::RegisteredInFactory<ImageViewer>
{
public:
    /** @brief Constructor. Just run s_bRegistered variable such that the class is
     * not discarded during compiler initialization. Using the power of static variables to ensure this **/
    ImageViewer() {
        DISABLE_WARNING_PUSH
                DISABLE_WARNING_UNREFERENCED_VARIABLE
        DISABLE_WARNING_UNUSED_VARIABLE
                s_bRegistered;
        DISABLE_WARNING_POP
    }
    /** @brief Static method to create instance of this class, returns a unique ptr of ImageViewer **/
    static std::unique_ptr<Base> CreateMethod() { return std::make_unique<ImageViewer>(); }
    /** @brief Name which is registered for this class. Same as 'ClassName' **/
    static std::string GetFactoryName() { return "ImageViewer"; }
    /** @brief Setup function called one during script creating prepare **/
    void setup() override;
    /** @brief update function called once per frame **/
    void update() override;
    /** @brief destroy function called before script deletion **/
    void onDestroy() override{
    }
    /** @brief set if this script should be drawn or not. */
    void setDrawMethod(VkRender::CRL_SCRIPT_DRAW_METHOD _drawMethod) override{ this->drawMethod = _drawMethod; }

    /** @brief draw function called once per frame **/
    void draw(CommandBuffer * commandBuffer, uint32_t i, bool b) override;

    /** @brief Get the type of script. Future extension if Scripts should behave differently **/
    VkRender::ScriptTypeFlags getType() override { return type; }
    VkRender::CRL_SCRIPT_DRAW_METHOD getDrawMethod() override {return drawMethod;}

    /** @brief public string to determine if this script should be attaced to an object,
     * create a new object or do nothing. Types: Render | None | Name of object in object folder **/
    VkRender::ScriptTypeFlags type = VkRender::CRL_SCRIPT_TYPE_RENDERER3D;
    VkRender::CRL_SCRIPT_DRAW_METHOD drawMethod = VkRender::CRL_SCRIPT_DONT_DRAW;
    SyclRenderer syclRenderer;

    std::unique_ptr<ImageView> imageView;
    std::unique_ptr<ImageView> imageView2;

};


#endif //MULTISENSE_VIEWER_MULTISENSE_H
