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

        imageComponent = std::make_unique<ImageComponent>(Utils::getTexturePath() / "icon_preview.png");
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        // Decide which pipeline to use
        m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
        m_renderPipelines->bindImage(*imageComponent);
    }


    void EditorImage::onEditorResize() {

        m_recreateOnNextImageChange = true;

    }


    void EditorImage::onFileDrop(const std::filesystem::path &path) {
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

        // If our renderpass was invalidated (by resize) recreate our pipelines

    }

    void EditorImage::onSceneLoad() {
        // Once we load a scene we need to create pipelines according to the objects specified in the scene.
        // For OBJModels we are alright with a default rendering pipeline (Phong lightining and stuff)
        // The pipelines also define memory handles between CPU and GPU. It makes more logical scenes if these attributes belong to the OBJModelComponent
        // But we need it accessed in the pipeline

    }


    void EditorImage::onUpdate() {
        // update Image when needed

    }

    void EditorImage::onRender(CommandBuffer &drawCmdBuffers) {
        m_renderPipelines->draw(drawCmdBuffers);

    }

    void EditorImage::onMouseMove(const MouseButtons &mouse) {

    }

    void EditorImage::onMouseScroll(float change) {

    }


}