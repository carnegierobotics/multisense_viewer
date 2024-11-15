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

    struct EditorGaussianViewerUI : public EditorUI {
        bool render3dgsImage = false;
        bool useImageFrom3DViewport = false;

        int32_t colorOption = 0;
        int radioButton = 0;

        // Constructor that copies everything from base EditorUI
        EditorGaussianViewerUI(const EditorUI &baseUI) : EditorUI(baseUI) {}
    };


    class EditorGaussianViewerLayer : public Layer {

    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }

        /** Called once per frame **/
        void onUIRender() override {

            // Set window position and size
            ImVec2 window_pos = ImVec2(5.0f, 55.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(m_editor->ui()->width - 5.0f,
                                        m_editor->ui()->height - 55.0f); // Size (width, height)


            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);

            // Create the parent window
            ImGui::Begin("EditorGaussianViewerLayer", nullptr, window_flags);
            auto imageUI = std::dynamic_pointer_cast<EditorGaussianViewerUI>(m_editor->ui());

            ImGui::RadioButton("3DGS", &imageUI->radioButton, 0);
            ImGui::SameLine();
            ImGui::RadioButton("2DGS", &imageUI->radioButton, 1);
            ImGui::SameLine();

            imageUI->render3dgsImage = ImGui::Button("Render 3DGS image");

            static bool toggle = true;
            ImGui::SameLine();
            ImGui::Checkbox("Toggle rendering", &toggle);
            if (toggle){
                imageUI->render3dgsImage = true;
            }

            /*
            ImGui::PushFont(m_editor->guiResources().font15);




            m_editor->ui()->saveRenderToFile = ImGui::Button("Save image");
            ImGui::Checkbox("Right view", &m_editor->ui()->gsRightView);
            ImGui::PopFont();
  */

            ImGui::End();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}

#endif //MULTISENSE_VIEWER_EDITORGAUSSIAN_VIEWER_LAYER
