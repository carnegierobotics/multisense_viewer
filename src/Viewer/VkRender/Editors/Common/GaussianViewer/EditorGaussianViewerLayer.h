//
// Created by magnus on 8/14/24.
//

#ifndef MULTISENSE_VIEWER_EDITORGAUSSIAN_VIEWER_LAYER
#define MULTISENSE_VIEWER_EDITORGAUSSIAN_VIEWER_LAYER


#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/ImGui/IconsFontAwesome6.h"
#include "Viewer/VkRender/Editors/EditorDefinitions.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {


    class EditorGaussianViewerLayer : public Layer {

    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }

        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles &handles) override {

            // Set window position and size
            ImVec2 window_pos = ImVec2(5.0f, 55.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(handles.editorUi->width - 5.0f, handles.editorUi->height - 55.0f); // Size (width, height)


            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

            // Create the parent window
            ImGui::Begin("EditorGaussianViewerLayer", nullptr, window_flags);

            ImGui::PushFont(handles.info->font15);
            handles.editorUi->render3DGSImage = ImGui::Button("Render 3DGS image");
            ImGui::PopFont();

            ImGui::End();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}

#endif //MULTISENSE_VIEWER_EDITORGAUSSIAN_VIEWER_LAYER
