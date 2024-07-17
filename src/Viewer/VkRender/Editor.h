//
// Created by magnus on 7/15/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR_H
#define MULTISENSE_VIEWER_EDITOR_H

#include <vk_mem_alloc.h>
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Core/VulkanRenderPass.h"
#include "Viewer/VkRender/ImGui/GuiManager.h"

namespace VkRender {
    class Renderer;

    class Editor {
    public:
        Editor() = delete;

        explicit Editor(const VulkanRenderPassCreateInfo &createInfo);

        ~Editor();

        void render(CommandBuffer &drawCmdBuffers);
        void update(bool updateGraph, float frametime, Input* input);

        std::vector<std::shared_ptr<VulkanRenderPass>> renderPasses;
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

        std::string editorTypeDescription;

    private:

    };
}

#endif //MULTISE{}NSE_VIEWER_EDITOR_H
