//
// Created by magnus on 1/15/25.
//

#ifndef MULTISENSE_VIEWER_DIFFRENTIABLE_RENDERER_LAYER
#define MULTISENSE_VIEWER_DIFFRENTIABLE_RENDERER_LAYER



#include "Viewer/Rendering/ImGui/Layer.h"


namespace VkRender
{
    class EditorDifferentiableRendererLayer : public Layer
    {
    public:
        void onAttach() override;

        void onDetach() override;

        void onUIRender() override;

        void onFinishedRender() override;
    };
}

#endif //MULTISENSE_VIEWER_DIFFRENTIABLE_RENDERER_LAYER
