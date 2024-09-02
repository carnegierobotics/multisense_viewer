//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORIMAGERLAYER
#define MULTISENSE_VIEWER_EDITORIMAGERLAYER

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/ImGui/IconsFontAwesome6.h"

namespace VkRender {


    class EditorImageLayer : public Layer {

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
            // Set window position and size
            ImVec2 window_pos = ImVec2(5.0f, 55.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(handles.editorUi->width - 5.0f,
                                        handles.editorUi->height - 55.0f); // Size (width, height)

            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBackground;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
            // Create the parent window
            ImGui::Begin("EditorImageLayer", nullptr, window_flags);
            //handles.editorUi->saveRenderToFile = ImGui::Button("Save image");
            //handles.editorUi->reloadPipeline = ImGui::Button("Reload pipeline");

            ImGui::End();


        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
}
#endif //MULTISENSE_VIEWER_EDITORIMAGERLAYER
