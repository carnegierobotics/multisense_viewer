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

            /*
            ImGui::PushFont(m_editor->guiResources().font15);

            m_editor->ui()->render3DGSImage = ImGui::Button("Render 3DGS image");

            static bool toggle = false;

            ImGui::RadioButton("3DGS image", &m_editor->ui()->render3dgsColor, 0);
            ImGui::RadioButton("Normals", &m_editor->ui()->render3dgsColor, 1);
            ImGui::RadioButton("Depth", &m_editor->ui()->render3dgsColor, 2);

            ImGui::Checkbox("Toggle rendering", &toggle);
            if (toggle){
                m_editor->ui()->render3DGSImage = true;
            }

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
