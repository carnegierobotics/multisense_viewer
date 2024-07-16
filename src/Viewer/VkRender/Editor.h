//
// Created by magnus on 7/15/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR_H
#define MULTISENSE_VIEWER_EDITOR_H

#include <vk_mem_alloc.h>
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/ImGui/GuiManager.h"

namespace VkRender {
    class Renderer;

    struct VkRenderEditorCreateInfo {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t x = 0;
        uint32_t y = 0;
        std::string editorTypeDescription;

        VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkImageLayout initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkImageLayout finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentStoreOp storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkImageView &colorImageView;
        VkImageView &depthImageView;
        std::shared_ptr<GuiResources> guiResources;

        VkRenderEditorCreateInfo(VkImageView &colorView,
                                 VkImageView &depthView, std::shared_ptr<GuiResources> guiRes) : colorImageView(colorView), depthImageView(depthView), guiResources(guiRes) {

        }

    };

    class Editor {
    public:
        Editor() = delete;

        explicit Editor(const VkRenderEditorCreateInfo &createInfo, RenderUtils &utils, Renderer &ctx);

        ~Editor();

        void render();

        EditorRenderPass depthRenderPass;
        EditorRenderPass objectRenderPass;
        EditorRenderPass uiRenderPass;
        RenderUtils &m_renderUtils;
        Renderer &m_context;

        uint32_t x = 0;
        uint32_t y = 0;

        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t applicationWidth = 0;
        uint32_t applicationHeight = 0;

        std::vector<VkFramebuffer> frameBuffers;

        std::unique_ptr<GuiManager> m_guiManager;

        glm::vec2 lastHoldPosition;
        std::string editorTypeDescription;

    private:
        void setupRenderPasses(EditorRenderPass *secondaryRenderPasses);

        void setupFrameBuffer();

    private:

        void setupUIRenderPass(const VkRenderEditorCreateInfo &createinfo, EditorRenderPass *secondaryRenderPasses);
    };
}

#endif //MULTISENSE_VIEWER_EDITOR_H
