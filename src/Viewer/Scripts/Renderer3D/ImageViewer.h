//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_MULTISENSE_H
#define MULTISENSE_VIEWER_MULTISENSE_H

#include "Viewer/Scripts/Private/ScriptBuilder.h"
#include "Viewer/Renderer/Components/CustomModels.h"

#ifdef SYCL_ENABLED
#include "Viewer/SYCL/GaussianRenderer.h"
#endif

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
    void onUIUpdate(VkRender::GuiObjectHandles *uiHandle) override;
    /** @brief destroy function called before script deletion **/
    void onDestroy() override{
    }

#ifdef SYCL_ENABLED
    std::unique_ptr<VkRender::GaussianRenderer> m_renderer;
#endif
    std::string splatEntity;

    std::unique_ptr<TextureVideo> m_syclRenderTarget;
    bool btn = false;
    bool render3dgsImage = false, render3dgs = false;
    void onWindowResize(const VkRender::GuiObjectHandles *uiHandle) override;

};


#endif //MULTISENSE_VIEWER_MULTISENSE_H
