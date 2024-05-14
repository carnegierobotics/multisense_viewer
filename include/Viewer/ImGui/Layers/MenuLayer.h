//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_MENULAYER_H
#define MULTISENSE_VIEWER_MENULAYER_H

#include "Viewer/ImGui/Layers/Layer.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {

    class MenuLayer : public VkRender::Layer {
    public:

        std::future<std::filesystem::path> loadFileFuture;
        std::future<std::filesystem::path> folderFuture;

        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }


        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles *handles) override {
            ImGui::BeginMainMenuBar();

            ImGui::EndMainMenuBar();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}
#endif //MULTISENSE_VIEWER_MENULAYER_H
