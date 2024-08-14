//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_PROPERTIESLAYER
#define MULTISENSE_VIEWER_PROPERTIESLAYER

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/ImGui/LayerUtils.h"
#include "Viewer/VkRender/Components/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"

namespace VkRender {


    class PropertiesLayer : public Layer {

    public:
        std::future<LayerUtils::LoadFileInfo> loadFileFuture;


        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }


        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles &handles) override {

            // Set window position and size
            ImVec2 window_pos = ImVec2(0.0f, 55.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(handles.editorUi->width, handles.editorUi->height); // Size (width, height)
            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoBringToFrontOnFocus;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

            // Create the parent window
            ImGui::Begin("PropertiesLayer", NULL, window_flags);

            ImGui::Text("Properties goes here");


            ImGui::End();
        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
}

#endif //MULTISENSE_VIEWER_PROPERTIESLAYER
