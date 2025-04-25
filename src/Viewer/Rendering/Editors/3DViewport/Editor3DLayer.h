//
// Created by magnus on 8/14/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR3DLAYER_H
#define MULTISENSE_VIEWER_EDITOR3DLAYER_H



#include <Viewer/Rendering/Editors/EditorIncludes.h>

#include "Viewer/Rendering/ImGui/Layer.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {
    enum class OutputTextureImageType { Color, Depth };

    enum class DepthColorOption : int32_t {
        None,
        Invert,
        Normalize,
        JetColormap,
        ViridisColormap
    };

    struct Editor3DViewportUI : EditorUI {
        bool renderFromViewpoint = false;
        bool saveNextFrame = false;
        // Image type selection
        OutputTextureImageType selectedImageType = OutputTextureImageType::Color;
        // Depth color option selection (only relevant if Depth is selected)
        DepthColorOption depthColorOption = DepthColorOption::None;
        bool reloadViewportShader = false;
        // Constructor that copies everything from base EditorUI
        explicit Editor3DViewportUI(const EditorUI &baseUI) : EditorUI(baseUI) {
        }
    };

    class Editor3DLayer : public Layer {
    public:
        /** Called once upon this object creation**/
        void onAttach() override ;

        /** Called after frame has finished rendered **/
        void onFinishedRender() override;


        /** Called once per frame **/
        void onUIRender() override ;

        /** Called once upon this object destruction **/
        void onDetach()
        override;
    };
}

#endif //MULTISENSE_VIEWER_EDITOR3DLAYER_H
