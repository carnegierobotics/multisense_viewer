//
// Created by magnus on 8/14/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR3DLAYER_H
#define MULTISENSE_VIEWER_EDITOR3DLAYER_H


#include <glm/gtc/quaternion.hpp>

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
        explicit Editor3DViewportUI(const EditorUI& baseUI) : EditorUI(baseUI) {
        }
    };


    class Editor3DLayer : public Layer {
    public:
        /** Called once upon this object creation**/
        void onAttach() override;

        /** Called after frame has finished rendered **/
        void onFinishedRender() override;


        /** Called once per frame **/
        void onUIRender() override;

        /** Called once upon this object destruction **/
        void onDetach() override;

        void drawSettingsTab();

        void decomposeTransform(
            const glm::mat4& m,
            glm::vec3& translation,
            glm::vec3& rotation,
            glm::vec3& scale) {
            // 1) Extract translation from the last column
            translation = glm::vec3(m[3]);

            // 2) Extract the columns' length, which correspond to the scale on each axis
            scale.x = glm::length(glm::vec3(m[0]));
            scale.y = glm::length(glm::vec3(m[1]));
            scale.z = glm::length(glm::vec3(m[2]));

            // 3) Remove scale from the matrix to isolate rotation
            glm::mat4 rotMat = m;
            if (scale.x != 0.0f) rotMat[0] /= scale.x;
            if (scale.y != 0.0f) rotMat[1] /= scale.y;
            if (scale.z != 0.0f) rotMat[2] /= scale.z;

            // 4) Convert the 3×3 rotation matrix to a quaternion...
            glm::quat q = glm::quat_cast(rotMat);

            // 5) …and then to Euler angles (in radians), finally converting to degrees
            glm::vec3 euler = glm::eulerAngles(q);
            rotation = glm::degrees(euler);
        }
    };
}

#endif //MULTISENSE_VIEWER_EDITOR3DLAYER_H
