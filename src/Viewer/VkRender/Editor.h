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

    struct VkRenderEditorCreateInfo{
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t x = 0;
        uint32_t y = 0;
        std::string description;
        void* mainUIContext;

        VkAttachmentLoadOp loadOp;
    };
    class Editor {
    public:
        Editor() = delete;

        explicit Editor(const VkRenderEditorCreateInfo& createInfo, RenderUtils &utils, Renderer &ctx);
        ~Editor();

        void render();

        EditorRenderPass depthRenderPass;
        EditorRenderPass objectRenderPass;
        EditorRenderPass uiRenderPass;
        RenderUtils &m_renderUtils;
        Renderer &m_context;

        std::unique_ptr<GuiManager> m_guiManager;

        uint32_t x = 0;
        uint32_t y = 0;

        uint32_t width = 0;
        uint32_t height = 0;

        std::vector<VkFramebuffer> frameBuffers;


    private:
        void setupRenderPasses(EditorRenderPass *secondaryRenderPasses);

        void setupUIRenderPass(EditorRenderPass *secondaryRenderPasses);

        void setupMainFramebuffer();

    private:


        struct {
            VkImage image;
            VkDeviceMemory mem;
            VkImageView view;
            VmaAllocation allocation;
        } m_depthStencil{};


        struct {
            VkImage image;
            VkDeviceMemory mem;
            VkImageView view;
            VmaAllocation allocation;
        } m_colorImage{};

        std::string description = "Editor data";

        void setupDepthStencil();

        void createColorResources();

        void setupUIRenderPass(const VkRenderEditorCreateInfo &createinfo, EditorRenderPass *secondaryRenderPasses);
    };
}

#endif //MULTISENSE_VIEWER_EDITOR_H
