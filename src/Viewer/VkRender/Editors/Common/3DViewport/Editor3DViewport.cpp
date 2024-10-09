//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Common/3DViewport/Editor3DViewport.h"

#include <Viewer/VkRender/Components/MaterialComponent.h>

#include "Editor3DLayer.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/RenderPipelines/UboGraphicsPipeline.h"

namespace VkRender {
    Editor3DViewport::Editor3DViewport(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("Editor3DLayer");
        // Grid and objects
        addUIData<Editor3DViewportUI>();
    }

    void Editor3DViewport::onEditorResize() {
        m_editorCamera.setPerspective(static_cast<float>(m_createInfo.width) / m_createInfo.height);
    }

    void Editor3DViewport::onSceneLoad(std::shared_ptr<Scene> scene) {
        // Once we load a scene we need to create pipelines according to the objects specified in the scene.
        // For OBJModels we are alright with a default rendering pipeline (Phong lightining and stuff)
        // The pipelines also define memory handles between CPU and GPU. It makes more logical scenes if these attributes belong to the OBJModelComponent
        // But we need it accessed in the pipeline
        m_editorCamera = Camera(m_createInfo.width, m_createInfo.height);
        m_activeCamera = m_editorCamera;
        m_activeScene = m_context->activeScene();
    }

    void Editor3DViewport::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);
        m_activeScene = m_context->activeScene();
        if (!m_activeScene)
            return;
        // Update model transforms:
        for (auto &pipeline: m_renderPipelines) {
            auto entity = Entity(static_cast<entt::entity>(pipeline.first), m_activeScene.get());
            auto &transform = entity.getComponent<TransformComponent>();
            pipeline.second->updateTransform(transform);
            pipeline.second->updateView(m_activeCamera);
            pipeline.second->update(m_context->currentFrameIndex());
        }
        m_activeScene->update(m_context->currentFrameIndex());
    }

    void Editor3DViewport::onComponentAdded(Entity entity, MeshComponent &meshComponent) {
        // add graphics pipeline for meshcomponent
        Log::Logger::getInstance()->info("Add meshcomponent for entity: {} in editor: {}",
                                         entity.getUUID().operator std::string(), getUUID().operator std::string());
        if (!m_renderPipelines.contains(entity)) {
            RenderPassInfo renderPassInfo{};
            renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
            renderPassInfo.renderPass = m_renderPass->getRenderPass();
            // Decide which pipeline to use
            if (meshComponent.usesUBOMesh()) {
                m_renderPipelines[entity] = std::make_unique<UboGraphicsPipeline>(*m_context, renderPassInfo);
            } else {
                m_renderPipelines[entity] = std::make_unique<DefaultGraphicsPipeline>(*m_context, renderPassInfo);
            }
            m_renderPipelines[entity]->bind(meshComponent);
        }
    }

    void Editor3DViewport::onRender(CommandBuffer &drawCmdBuffers) {
        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);

        for (auto &val: m_renderPipelines | std::views::values) {
            val->draw(drawCmdBuffers);
        }
    }

    void Editor3DViewport::initializeMaterial(MaterialComponent& material) {
        /*
         *
        *If multiple materials use the same descriptor set layout, you can cache and reuse them:

        std::unordered_map<DescriptorSetLayoutKey, VkDescriptorSetLayout> descriptorSetLayoutCache;
        DescriptorSetLayoutKey: Define a key based on the bindings and types of descriptors.
        Reuse: Before creating a new descriptor set layout, check if an identical one already exists in the cache.
         **/
        // Define descriptor set layout bindings based on the material's properties
        std::vector<VkDescriptorSetLayoutBinding> bindings;

        // Example: if the material uses a uniform buffer
        if (material.usesUniformBuffer) {
            VkDescriptorSetLayoutBinding uboLayoutBinding = {};
            uboLayoutBinding.binding = 0;
            uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            uboLayoutBinding.descriptorCount = 1;
            uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            uboLayoutBinding.pImmutableSamplers = nullptr;
            bindings.push_back(uboLayoutBinding);
        }

        // Example: if the material uses a combined image sampler (texture)
        if (material.usesTexture) {
            VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
            samplerLayoutBinding.binding = 1;
            samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            samplerLayoutBinding.descriptorCount = 1;
            samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            samplerLayoutBinding.pImmutableSamplers = nullptr;
            bindings.push_back(samplerLayoutBinding);
        }

        // Create the descriptor set layout
        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(m_context->vkDevice().m_LogicalDevice, &layoutInfo, nullptr, &material.descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }

        // Allocate and write to the descriptor set
        //allocateAndWriteDescriptorSet(material);
    }

    void Editor3DViewport::collectRenderCommands(std::unordered_map<std::shared_ptr<GraphicsPipeline>, std::vector<RenderCommand>>& renderGroups) {
        auto view = m_activeScene->getRegistry().view<MeshComponent, MaterialComponent, TransformComponent>();

        for (auto entity : view) {
            auto& meshComponent = view.get<MeshComponent>(entity);
            auto& materialComponent = view.get<MaterialComponent>(entity);
            auto& transformComponent = view.get<TransformComponent>(entity);

            // Ensure the material's descriptor set layout is initialized
            if (materialComponent.descriptorSetLayout == VK_NULL_HANDLE) {
                initializeMaterial(materialComponent);
            }

            // Create or retrieve pipeline
            PipelineKey key = { materialComponent.renderMode, materialComponent.shaderName, materialComponent.descriptorSetLayout };
            auto pipeline = m_pipelineManager.getOrCreatePipeline(key, m_renderPass->getRenderPass());

            // Create render command
            RenderCommand command;
            command.entity = Entity(entity, m_activeScene.get());
            command.pipeline = pipeline;
            command.mesh = &meshComponent;
            command.material = &materialComponent;
            command.transform = &transformComponent;

            // Add to render group
            renderGroups[pipeline].push_back(command);
        }
    }

    void Editor3DViewport::onRenderDepthOnly(CommandBuffer &drawCmdBuffers) {
        for (auto &pipeline: m_depthOnlyRenderPipelines) {
            pipeline.second->draw(drawCmdBuffers);
        }
    }

    void Editor3DViewport::onEntityDestroyed(entt::entity entity) {
        Entity e(entity, m_activeScene.get());
        if (e.hasComponent<MeshComponent>()) {
            if (m_renderPipelines.contains(e.getComponent<MeshComponent>().getUUID()))
                m_renderPipelines.erase(e.getComponent<MeshComponent>().getUUID());
        }
    }

    void Editor3DViewport::onMouseMove(const MouseButtons &mouse) {
        if (ui()->hovered && mouse.left && !ui()->resizeActive) {
            m_editorCamera.rotate(mouse.dx, mouse.dy);
        }
        m_activeScene->onMouseEvent(mouse);
    }

    void Editor3DViewport::onMouseScroll(float change) {
        if (ui()->hovered)
            m_editorCamera.setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
        m_activeScene->onMouseScroll(change);
    }
};

