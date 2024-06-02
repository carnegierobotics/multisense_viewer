//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_BASE_H
#define MULTISENSE_VIEWER_BASE_H


#include "Viewer/Tools/Macros.h"
#include "Viewer/ImGui/Layers/Layer.h"
#include "Viewer/Core/CommandBuffer.h"

namespace VkRender {
    /**
     * @brief Base class for scripts that can be attached to renderer. See @refitem Example for how to implement a script.
     */
    class Base {
    public:
        Renderer* m_context;
        virtual ~Base() = default;

        DISABLE_WARNING_PUSH
        DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
        DISABLE_WARNING_EMPTY_BODY

        /**@brief Pure virtual function called only once when VK is ready to render*/
        virtual void setup() {
        }

        /**@brief Pure virtual function called once every frame*/
        virtual void update() {
        }

        /**@brief Pure virtual function called each frame*/
        virtual void onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {
        }

        /**@brief Virtual function called when resize event is triggered from the platform os*/
        virtual void onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {
        }

        /**@brief Called once script is requested for deletion */
        virtual void onDestroy() {
        }

        void windowResize(const VkRender::GuiObjectHandles *uiHandle) {
            onWindowResize(uiHandle);
        }

        void uiUpdate(VkRender::GuiObjectHandles *uiHandle) {
            onUIUpdate(uiHandle);
        }
        DISABLE_WARNING_POP

        void updateUniformBufferData() {
                update();
        }

        void createUniformBuffers() {
            setup();
        }

        /**@brief Call to delete the attached script. */
        void onDestroyScript() {
            onDestroy();
        }

    private:

    };
};
#endif //MULTISENSE_VIEWER_BASE_H
