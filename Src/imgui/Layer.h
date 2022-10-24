//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_LAYER_H
#define MULTISENSE_LAYER_H

#define IMGUI_INCLUDE_IMGUI_USER_H
#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS

#include "imgui.h"
#include <array>
#include "map"
#include "unordered_map"
#include "string"
#include "MultiSense/Src/Core/Definitions.h"
#include "MultiSense/Src/Core/KeyInput.h"
#include <memory>

namespace VkRender {

    /**
     * @brief A UI Layer drawn by \refitem GuiManager.
     * To add an additional UI layer see \refitem LayerExample.
     */
    class Layer {

    public:

        virtual ~Layer() = default;

        /** @brief
         * Pure virtual must be overridden.
         * Called ONCE after UI have been initialized
         */
        virtual void onAttach() = 0;

        /** @brief
         * Pure virtual must be overridden.
         * Called ONCE before UI objects are destroyed
         */
        virtual void onDetach() = 0;

        /**
         * @brief Pure virtual must be overridden.
         * Called per frame, but before each script (\refitem Example) is updated
         * @param handles a UI object handle to collect user input
         */
        virtual void onUIRender(GuiObjectHandles *handles) = 0;

        /**
         * @brief Pure virtual must be overridden.
         * Called after draw command have been recorded, but before this frame has ended.
         * Can be used to prepare for next frame for instance
         */
        virtual void onFinishedRender() = 0;
    };
};

#endif //MULTISENSE_LAYER_H
