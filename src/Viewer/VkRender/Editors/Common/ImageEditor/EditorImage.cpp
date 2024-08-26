//
// Created by mgjer on 18/08/2024.
//

#include "Viewer/VkRender/Editors/Common/ImageEditor/EditorImage.h"
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/RenderPipelines/RenderBase.h"

namespace VkRender {
    EditorImage::EditorImage(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("EditorImageLayer");

        /*
        imageComponent = std::make_unique<ImageComponent>(Utils::getTexturePath() / "icon_preview.png");

        // Decide which pipeline to use
        m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
        m_renderPipelines->bindImage(*imageComponent);
        */

        /*
        // Create Texture for our image view
        VkImageCreateInfo imageCreateInfo;
        VkImageViewCreateInfo imageViewCreateInfo;

        VulkanImageCreateInfo vulkanImageCreateInfo(m_context->vkDevice(), m_context->allocator(), imageCreateInfo, imageViewCreateInfo);
        std::shared_ptr<VulkanImage> image = std::make_shared<VulkanImage>(vulkanImageCreateInfo);
        */

    }

    void EditorImage::onEditorResize() {
        m_recreateOnNextImageChange = true;
    }

    void EditorImage::onFileDrop(const std::filesystem::path &path) {
        /*
        if (m_recreateOnNextImageChange){
            std::filesystem::path filePath = imageComponent->getTextureFilePath();
            imageComponent = std::make_unique<ImageComponent>(filePath);
            RenderPassInfo renderPassInfo{};
            renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
            renderPassInfo.renderPass = m_renderPass->getRenderPass();
            // Decide which pipeline to use
            m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
            m_renderPipelines->bindImage(*imageComponent);
            m_recreateOnNextImageChange = false;
        }

        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if (extension == ".png" || extension == ".jpg") {
            imageComponent = std::make_unique<ImageComponent>(path);
            m_renderPipelines->bindImage(*imageComponent);
        }
        */
    }

    void EditorImage::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_activeScene = m_context->activeScene();
    }


    void EditorImage::onPipelineReload() {
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = m_context->sharedEditorData().depthFrameBuffer[m_context->sharedEditorData().selectedUUIDContext.operator*()].depthImage; // TODO get this from the Editor3D Viewport then I think it is solved
        m_texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
        m_depthImagePipeline = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
        m_depthImagePipeline->bindTexture(m_texture);
    }

    void EditorImage::onUpdate() {

        if (m_context->sharedEditorData().selectedUUIDContext && !m_depthImagePipeline) {
            RenderPassInfo renderPassInfo{};
            renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
            renderPassInfo.renderPass = m_renderPass->getRenderPass();
            VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
            textureCreateInfo.image = m_context->sharedEditorData().depthFrameBuffer[m_context->sharedEditorData().selectedUUIDContext.operator*()].depthImage; // TODO get this from the Editor3D Viewport then I think it is solved
            m_texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
            m_depthImagePipeline = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
            m_depthImagePipeline->bindTexture(m_texture);
        }

        //ui().saveRenderToFile = m_createInfo.sharedUIContextData->newFrame;
        if (m_createInfo.sharedUIContextData->m_selectedEntity){
            ui().renderToFileName = "scene_0000/disparity/" +  m_createInfo.sharedUIContextData->m_selectedEntity.getComponent<TagComponent>().Tag;
            ui().renderToFileName.replace_extension(".png");
        }

        auto view = m_activeScene->getRegistry().view<CameraComponent>();
        for (auto entity: view) {
            auto e = Entity(entity, m_activeScene.get());
            if (e == m_createInfo.sharedUIContextData->m_selectedEntity) {
                auto &cameraComponent = view.get<CameraComponent>(entity);
                m_activeCamera = std::make_shared<Camera>(cameraComponent());
            }
        }


        if (m_depthImagePipeline && m_activeCamera) {

            m_depthImagePipeline->updateView(m_activeCamera.operator*());
            m_depthImagePipeline->update(m_context->currentFrameIndex());
        }
        m_activeScene->update(m_context->currentFrameIndex());
        // update Image when needed

        // Check if any 3D viewport is rendering depth

        // We want to attach the depth only image

        // Update it from the latest render
    }

    void EditorImage::onRender(CommandBuffer &drawCmdBuffers) {


        if (m_depthImagePipeline)
            m_depthImagePipeline->draw(drawCmdBuffers);
    }

    void EditorImage::onMouseMove(const MouseButtons &mouse) {
    }

    void EditorImage::onMouseScroll(float change) {
    }

}