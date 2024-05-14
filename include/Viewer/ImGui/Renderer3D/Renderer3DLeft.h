//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_RENDERER3DLEFT_H
#define MULTISENSE_VIEWER_RENDERER3DLEFT_H

#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/ImGui/LayerUtils.h"

#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Entity.h"
#include "Viewer/Renderer/Components/GLTFModelComponent.h"
#include "Viewer/Renderer/Components/SkyboxGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components/OBJModelComponent.h"
#include "Viewer/Renderer/Components/CameraGraphicsPipelineComponent.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/
namespace VkRender {
    class Renderer3DLeft : public Layer {
    public:
        std::future<std::filesystem::path> loadFileFuture;


        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }



        /** Called once per frame **/
        void onUIRender(GuiObjectHandles *handles) override {
            if (!handles->renderer3D)
                return;


        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
};
DISABLE_WARNING_POP


#endif //MULTISENSE_VIEWER_RENDERER3DLEFT_H
