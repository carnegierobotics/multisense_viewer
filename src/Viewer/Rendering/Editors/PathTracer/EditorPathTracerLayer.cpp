//
// Created by magnus on 2/4/25.
//

#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayer.h"

#include <Viewer/Rendering/ImGui/IconsFontAwesome6.h>
#include <Viewer/Rendering/ImGui/LayerUtils.h>

#include "EditorPathTracer.h"
#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayerUI.h"
#include "Viewer/Rendering/Editors/Editor.h"

namespace VkRender {
    /** Called once upon this object creation**/
    void EditorPathTracerLayer::onAttach() {
    }


    /** Called once per frame **/
    void EditorPathTracerLayer::onUIRender() {
        // Set window position and size
        // Set window position and size
        ImVec2 window_pos = ImVec2(m_editor->ui()->layoutConstants.uiXOffset, 0.0f); // Position (x, y)
        ImVec2 window_size = ImVec2(m_editor->ui()->width - window_pos.x,
                                    m_editor->ui()->height - window_pos.y); // Size (width, height)

        // Set window flags to remove decorations
        ImGuiWindowFlags window_flags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBackground;

        // Set next window position and size
        ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
        // Create the parent window
        ImGui::Begin("EditorPathTracerLayer", nullptr, window_flags);

        static int currentTab = 0; // keep outside the loop
        const VerticalIconTab kTabs[] =
        {
            {ICON_FA_TV, "Renderer Settings", [this] { drawRendererSettingsTab(); }, 300.0f},
            {ICON_FA_BUG, "Debug", [this] {drawDebugViewTab(); }, 300.0f},
        };
        LayerUtils::drawVerticalIconTabs(kTabs, IM_ARRAYSIZE(kTabs), currentTab, m_editor);

        ImGui::End();
    }

    void EditorPathTracerLayer::drawDebugViewTab() {
        auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_editor->ui());
        ImGui::Text("Show BVH:"); ImGui::SameLine();

        ImGui::Checkbox("##Show BVH", &imageUI->showBVH);
        if (imageUI->showBVH) {
            ImGui::Spacing(); ImGui::SameLine();
            ImGui::Text("TLAS:"); ImGui::SameLine(); ImGui::Checkbox("##Show TLAS", &imageUI->showTLAS);
            ImGui::Spacing(); ImGui::SameLine();
            ImGui::Text("BLAS:"); ImGui::SameLine(); ImGui::Checkbox("##Show BLAS", &imageUI->showBLAS);
            ImGui::Spacing(); ImGui::SameLine();
            ImGui::Text("Min Level:"); ImGui::SameLine();
            ImGui::SliderInt("##BVH Depth Min", &imageUI->bvhLevelMin, 0, 30);
            ImGui::Spacing(); ImGui::SameLine();
            ImGui::Text("Max Level:"); ImGui::SameLine();
            ImGui::SliderInt("##BVH Depth Max", &imageUI->bvhLevelMax, 1, 30);
            ImGui::Spacing(); ImGui::SameLine();

        }

    }

    void EditorPathTracerLayer::drawRendererSettingsTab() {
        auto imageUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_editor->ui());
        imageUI->reloadRenderer = ImGui::Button("Upload scene");
        ImGui::Checkbox("Render", &imageUI->render);
        ImGui::Checkbox("To Viewport", &imageUI->renderToViewport);
        // --- Device selector as a dropdown ---
        ImGui::Text("Compute Device:");
        ImGui::SetNextItemWidth(100.0f);
        {
            // the labels in the dropdown
            const char* deviceNames[] = {"CPU", "GPU"};
            // we keep an int for Combo, matching our enum values
            int current = static_cast<int>(imageUI->selectedDevice);
            if (ImGui::Combo("##ComputeDevice", &current, deviceNames, IM_ARRAYSIZE(deviceNames))) {
                imageUI->selectedDevice = static_cast<SYCLDeviceType>(current);
            }
        }


        // --- New: Photon count slider ---
        // Photon count slider in steps of 10
        // We scale down by 10 for the slider then re-scale back up so it only ever hits multiples of 10.
        ImGui::Text("Photon Count:");
        ImGui::SetNextItemWidth(150.0f);
        {
            // slider from exponent 0..7
            int expo = imageUI->photonExponent;
            if (ImGui::SliderInt("##PhotonExp", &expo, 0, 7, "%d")) {
                imageUI->photonExponent = expo;
                imageUI->photonCount = static_cast<int>(std::pow(10, expo));
            }
            ImGui::Text("%d", imageUI->photonCount);
        }

        /*
            // Prepare dropdown items
            const char *kernels[PathTracer::KERNEL_TYPE_COUNT];

            for (int i = 0; i < PathTracer::KERNEL_TYPE_COUNT; ++i) {
                kernels[i] = PathTracer::KernelTypeToString(static_cast<PathTracer::KernelType>(i));
            }
            // Render ImGui combo box
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::Combo("##Render Kernel", &imageUI->selectedKernelIndex, kernels,
                             PathTracer::KERNEL_TYPE_COUNT)) {
                // Update the kernel based on selection
            }
            imageUI->kernel = static_cast<PathTracer::KernelType>(imageUI->selectedKernelIndex);


            // Dropdown for selecting render kernel
            const char *selections[] = {"CPU", "GPU"}; // TODO This should come from selectSyclDevices
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::Combo("##Select Device Type", &imageUI->selectedDeviceIndex, selections,
                             IM_ARRAYSIZE(selections))) {
                imageUI->switchKernelDevice = true;
                imageUI->kernelDevice = selections[imageUI->selectedDeviceIndex];
            }


            if (ImGui::Button("Clear image memory")) {
                imageUI->clearImageMemory = true;
            }

            const int sliderMin = 1;
            const int sliderMax = 10000000;

            ImGui::SetNextItemWidth(150);
            if (ImGui::SliderInt("PhotonCount", &imageUI->photonCount, sliderMin, sliderMax, "%d",
                                 ImGuiSliderFlags_Logarithmic)) {
                // Normalize to the nearest 10,000 and ensure it's at least 1000
                //imageUI->photonCount = std::max((imageUI->photonCount + 5000) / 10000 * 10000, sliderMin);
            }
                    ImGui::SetNextItemWidth(100);
            if (ImGui::SliderInt("Light Bounces", &imageUI->numBounces, 1, 100)) {
                imageUI->clearImageMemory = true;
            };
                    ImGui::SetNextItemWidth(100);
            ImGui::SliderFloat("Gamma", &imageUI->shaderSelection.gammaCorrection, 0, 8);

                    imageUI->saveImage = ImGui::Button("Save");


            // new row
            auto *editor = dynamic_cast<EditorPathTracer *>(m_editor);
            if (editor && editor->getRenderInformation())
                ImGui::Text("Frame Number: %u", editor->getRenderInformation()->frameID);

            ImGui::SameLine();
            ImGui::Checkbox("Apply Beta dist.", &imageUI->applyBetaContribution);

            */
    }

    /** Called once upon this object destruction **/
    void EditorPathTracerLayer::onDetach() {
    }

    void EditorPathTracerLayer::onFinishedRender() {
    }
}
