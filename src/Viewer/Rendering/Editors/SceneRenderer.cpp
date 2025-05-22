//
// Created by mgjer on 30/09/2024.
//


#include "Viewer/Rendering/Editors/SceneRenderer.h"

#include <Viewer/Rendering/Components/LightSourceComponent.h>
#include <Viewer/Rendering/Core/VulkanShaderModule.h>

#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/Components/MeshComponent.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"

#include "Viewer/Application/Application.h"

#include "Viewer/Scenes/Entity.h"

#include "Viewer/Rendering/MeshInstance.h"
#include "Viewer/Assets/ShaderLoader.h"


namespace VkRender {
    void MeshInstance::ensureInstanceBuffer(VkDeviceSize size, VulkanDevice &dev) {
        if (!instanceBuffer) {
            dev.createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, // must include VERTEX_BUFFER_BIT!
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, // simplest for frequently-updated data
                instanceBuffer,
                size,
                nullptr);
        }
        if (instanceBuffer->m_size >= size) {
            dev.createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, // must include VERTEX_BUFFER_BIT!
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, // simplest for frequently-updated data
                instanceBuffer,
                size,
                nullptr);
        };
    }

    SceneRenderer::SceneRenderer(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        m_renderToOffscreen = true;
        m_activeCamera = std::make_shared<BaseCamera>(static_cast<float>(m_createInfo.width) / m_createInfo.height);
        descriptorRegistry.createManager(DescriptorManagerType::MVP, m_context->vkDevice());
        descriptorRegistry.createManager(DescriptorManagerType::Transform, m_context->vkDevice());
        descriptorRegistry.createManager(DescriptorManagerType::MaterialData, m_context->vkDevice());
        descriptorRegistry.createManager(DescriptorManagerType::MaterialSampler, m_context->vkDevice());
        descriptorRegistry.createManager(DescriptorManagerType::DynamicCameraGizmo, m_context->vkDevice());

        m_meshResourceManager = std::make_unique<MeshResourceManager>(m_context);

        createGlobalBuffers();
        createGlobalPipelineLayout();

        /*
        VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = 2;
        vkCreateQueryPool(m_context->vkDevice().m_LogicalDevice, &qp, nullptr, &m_timestampPool);
        */
    }

    void SceneRenderer::onEditorResize() {
    }

    SceneRenderer::~SceneRenderer() {

        vkDestroyPipelineLayout(m_context->vkDevice().m_LogicalDevice, m_pipelineLayout, nullptr);
    }

    void SceneRenderer::createGlobalBuffers() {
        const uint32_t frames = m_context->swapChainBuffers().size();
        auto &dev = m_context->vkDevice();

        m_globalUbo.init<GlobalUBO>(dev, frames, 1, "GlobalUBO");
        m_transformSsbo.init<InstanceTransform>(dev, frames, kMaxEntities, "TransformSSBO");
        m_materialSsbo.init<MaterialBufferObject>(dev, frames, kMaxMaterials, "MaterialSSBO");

        std::vector<VkWriteDescriptorSet> writes;
        // 2. Descriptor‑set layouts (from your registry)
        auto &mvpMgr = descriptorRegistry.getManager(DescriptorManagerType::MVP);
        auto &transformMgr = descriptorRegistry.getManager(DescriptorManagerType::Transform);
        auto &materialMgr = descriptorRegistry.getManager(DescriptorManagerType::MaterialData);


        m_globalSets.resize(frames);
        m_transformSets.resize(frames);
        m_materialSets.resize(frames);

        // 3. Build Write structures → ask manager for a cached set per frame
        for (uint32_t f = 0; f < frames; ++f) {
            // ---------- Global set ------------------------------------------------
            VkDescriptorBufferInfo gInfo = m_globalUbo.info(f, sizeof(GlobalUBO));
            VkWriteDescriptorSet gWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            gWrite.dstBinding = 0;
            gWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            gWrite.descriptorCount = 1;
            gWrite.pBufferInfo = &gInfo;
            m_globalSets[f] = mvpMgr.getOrCreateDescriptorSet({gWrite});

            // ---------- Transform set --------------------------------------------
            VkDescriptorBufferInfo tInfo = m_transformSsbo.info(f, sizeof(InstanceTransform) * kMaxEntities);
            VkWriteDescriptorSet tWrite = gWrite;
            tWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            tWrite.pBufferInfo = &tInfo;
            m_transformSets[f] = transformMgr.getOrCreateDescriptorSet({tWrite});

            // ---------- Material set ---------------------------------------------
            VkDescriptorBufferInfo mInfo = m_materialSsbo.info(f, sizeof(MaterialBufferObject) * kMaxMaterials);
            VkWriteDescriptorSet mWrite = tWrite;
            mWrite.pBufferInfo = &mInfo;
            m_materialSets[f] = materialMgr.getOrCreateDescriptorSet({mWrite});
        }
    }

    void SceneRenderer::createGlobalPipelineLayout() {
        auto &dev = m_context->vkDevice();

        auto &mvpMgr = descriptorRegistry.getManager(DescriptorManagerType::MVP);
        auto &transformMgr = descriptorRegistry.getManager(DescriptorManagerType::Transform);
        auto &materialMgr = descriptorRegistry.getManager(DescriptorManagerType::MaterialData);
        auto &samplerMgr = descriptorRegistry.getManager(DescriptorManagerType::MaterialSampler);


        std::array<VkDescriptorSetLayout, 4> layouts = {
            {
                mvpMgr.getDescriptorSetLayout(),
                transformMgr.getDescriptorSetLayout(),
                materialMgr.getDescriptorSetLayout(),
                samplerMgr.getDescriptorSetLayout(),
            }
        };

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(BatchPC);

        VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        info.setLayoutCount = static_cast<uint32_t>(layouts.size());
        info.pSetLayouts = layouts.data();
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &pcRange;

        vkCreatePipelineLayout(dev.m_LogicalDevice, &info, nullptr, &m_pipelineLayout);
    }

    void SceneRenderer::onSceneLoad(std::shared_ptr<Scene> scene) {
        // Once we load a scene we need to create pipelines according to the objects specified in the scene.
        // For OBJModels we are alright with a default rendering pipeline (Phong lightining and stuff)
        // The pipelines also define memory handles between CPU and GPU. It makes more logical scenes if these attributes belong to the OBJModelComponent
        // But we need it accessed in the pipeline
        m_activeScene = m_context->activeScene();
    }

    void SceneRenderer::onUpdate() {
        m_activeScene = m_context->activeScene();
    }

    void SceneRenderer::onRender(CommandBuffer &commandBuffer) {
        // 0) Reset stats & timers
        m_stats.reset();
        CpuTimer tFrame;
        tFrame.start();

        uint32_t fIdx = commandBuffer.getActiveFrameIndex();
        VkCommandBuffer cb = commandBuffer.getActiveBuffer();

        // 1) Single-pass: build batches, transforms SSBO data, material SSBO data
        m_matIndices.clear();
        /* --------------------------------------------------------------------------
         * 1. first pass – build batches and collect matrices per-batch
         * -------------------------------------------------------------------------- */
        std::unordered_map<const MaterialComponent *, uint32_t> matIndices;
        m_batches.clear();

        auto view = m_activeScene->getRegistry().view<MeshComponent, TransformComponent>();
        for (auto [e, mc, tc]: view.each()) {
            Entity ent(e, m_activeScene.get());
            if (!isEntityTreeVisible(ent)) continue;

            /* ---- pipeline key / mesh instance ------------------------------------ */
            auto meshData = MeshManager::instance().getMeshData(mc);
            if (!meshData)
                continue;
            auto meshInst = m_meshResourceManager->getMeshInstance(
                mc.getCacheIdentifier(), meshData, mc.meshDataType());
            if (!meshInst)
                continue;

            auto matInst = getMaterialInstance(ent);
            PipelineKey key = makeKey(mc, *meshInst, matInst.get(), ent);

            /* ---- batch ----------------------------------------------------------- */
            auto &batch = m_batches[key];
            if (!batch.started) {
                batch.started = true;
                batch.mesh = meshInst;
                batch.material = matInst;
                batch.key = key;
            }
            batch.cpuInstanceTransform.push_back({tc.getTransform()}); // store locally
            if (ent.hasComponent<MaterialComponent>()) {
                auto &mcComp = ent.getComponent<MaterialComponent>();
                batch.cpuInstanceMaterial.push_back({makeMaterialBuffer(mcComp)}); // store locally
            }
        }

        /* --------------------------------------------------------------------------
         * 2. second pass – flatten matrices in *batch draw* order
         * -------------------------------------------------------------------------- */
        std::vector<InstanceTransform> transforms; // contiguous upload buffer
        transforms.reserve(view.size_hint()); // exact upper bound
        uint32_t globalCursor = 0;

        for (auto &kv: m_batches) {
            auto &batch = kv.second;

            batch.transformBase = globalCursor; // slice start in SSBO

            transforms.insert(transforms.end(),
                              batch.cpuInstanceTransform.begin(),
                              batch.cpuInstanceTransform.end());

            globalCursor += static_cast<uint32_t>(batch.cpuInstanceTransform.size());
            batch.instanceCount = static_cast<uint32_t>(batch.cpuInstanceTransform.size());
        }
        /* --------------------------------------------------------------------------
         * 2. second pass – flatten matrices in *batch draw* order
         * -------------------------------------------------------------------------- */
        uint32_t globalMaterialCursor = 0;
        std::vector<MaterialBufferObject> materials;
        materials.reserve(kMaxMaterials);

        for (auto &kv: m_batches) {
            auto &batch = kv.second;

            batch.materialBase = globalMaterialCursor; // slice start in SSBO

            materials.insert(materials.end(),
                              batch.cpuInstanceMaterial.begin(),
                              batch.cpuInstanceMaterial.end());

            globalMaterialCursor += static_cast<uint32_t>(batch.cpuInstanceMaterial.size());
        }

        /* --------------------------------------------------------------------------
         * 3. upload SSBOs
         * -------------------------------------------------------------------------- */
        m_transformSsbo.upload(m_context->vkDevice(), fIdx,
                               transforms.data(), transforms.size(),
                               sizeof(InstanceTransform));

        m_materialSsbo.upload(m_context->vkDevice(), fIdx,
                              materials.data(), materials.size(),
                              sizeof(MaterialBufferObject));

        // 3) Upload Global UBO
        GlobalUBO gUbo{};
        if (auto cam = m_activeCamera.lock()) {
            gUbo.view = cam->matrices.view;
            gUbo.proj = cam->matrices.projection;
            gUbo.cameraPos = cam->matrices.position;
        }
        uint32_t lcount = 0;
        auto lightView = m_activeScene->getRegistry().view<LightSourceComponent, TransformComponent>();
        for (auto [e, ls, tr]: lightView.each()) {
            if (lcount < kMaxLights) {
                gUbo.lightPos[lcount++] = glm::vec4(tr.getPosition(), 1.0f);
            }
        }
        if (lcount == 0) {
            gUbo.lightPos[0] = glm::vec4(5.0f, 5.05, 5.0f, 1.0f);
        }
        gUbo.numLights = float(lcount);
        m_globalUbo.upload(m_context->vkDevice(), fIdx,
                           &gUbo, 1, sizeof(GlobalUBO));

        // 4) Bind descriptor sets 0..2 once
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_pipelineLayout, 0, 1, &m_globalSets[fIdx], 0, nullptr);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_pipelineLayout, 1, 1, &m_transformSets[fIdx], 0, nullptr);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_pipelineLayout, 2, 1, &m_materialSets[fIdx], 0, nullptr);

        // 5) Issue draw calls per batch
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        renderPassInfo.debugName = "SceneRenderer::";

        for (auto &kv: m_batches) {
            auto &batch = kv.second;
            batch.mesh->instanceCount = batch.instanceCount;

            // We dont support rendering a material so if it doesn't exist create a dummy material that is empty
            if (!batch.material) {
                batch.material = std::make_shared<MaterialInstance>();
                // 1 Load Shader code
                auto vsSPV = assetManager()->get<SPIRVAsset>("BlinnPhongShader.vert");
                auto fsSPV = assetManager()->get<SPIRVAsset>( "NoMaterial.frag");
                // 2) Wrap into a GPU resource
                VulkanShaderModuleCreateInfo vertexShaderCreateInfo(m_context->vkDevice(), vsSPV, VK_SHADER_STAGE_VERTEX_BIT,
                                                                    "BlinnPhongShader.vert");
                VulkanShaderModuleCreateInfo fragmentShaderCreateInfo(m_context->vkDevice(), fsSPV,
                                                                      VK_SHADER_STAGE_FRAGMENT_BIT,
                                                                      "BlinnPhongShader.frag");
                // 3) Later in pipeline creation:

                // 3) Ask the GPU cache for shared modules:
                auto vsModule = cache()->shaderModules.get(vertexShaderCreateInfo);
                auto fsModule = cache()->shaderModules.get(fragmentShaderCreateInfo);

                batch.material->addShader(vsModule);
                batch.material->addShader(fsModule);
            }
            // pipeline (reuse or create)
            auto pipeline = m_pipelineManager.getOrCreatePipeline(
                batch.key,
                makePipelineInfo(batch.material.get()),
                renderPassInfo,
                m_pipelineLayout,
                m_context);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline->getPipeline());

            // bind sampler set at set=3
            if (batch.material && batch.material->baseColorTexture) {
                auto samplerSet = buildMaterialSamplerSet(batch.material);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        m_pipelineLayout, 3, 1, &samplerSet[DescriptorManagerType::MaterialSampler], 0,
                                        nullptr);
            }

            // push constants
            BatchPC pc{batch.transformBase, batch.materialBase};
            vkCmdPushConstants(cb, m_pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);

            RenderCommand renderCommand;
            renderCommand.pipeline = pipeline;
            renderCommand.meshInstance = batch.mesh.get();
            renderCommand.materialInstance = batch.material.get();
            // vertex & index bindings
            bindResourcesAndDraw(commandBuffer, renderCommand);

            m_stats.instanceTotal += batch.mesh->instanceCount;
        }

        // 6) Stats and end
        m_stats.drawCalls = static_cast<uint32_t>(m_batches.size());
        debugPrintStats();
    }


    void SceneRenderer::bindResourcesAndDraw(const CommandBuffer &commandBuffer,
                                             const RenderCommand &cmd) {
        VkCommandBuffer cb = commandBuffer.getActiveBuffer();

        /* ── 1. pipeline ─────────────────────────────────────────────── */
        vkCmdBindPipeline(cb,
                          VK_PIPELINE_BIND_POINT_GRAPHICS,
                          cmd.pipeline->getPipeline());

        /* ── 3. vertex buffer; **no** instance buffer any more ───────── */
        VkBuffer vbo = cmd.meshInstance->vertexBuffer->m_buffer;
        VkDeviceSize offs = 0;
        vkCmdBindVertexBuffers(cb, 0, 1, &vbo, &offs); // binding 0 only

        /* optional index buffer */
        if (cmd.meshInstance->indexBuffer) {
            vkCmdBindIndexBuffer(cb,
                                 cmd.meshInstance->indexBuffer->m_buffer,
                                 0, VK_INDEX_TYPE_UINT32);
        }

        /* ── 4. draw ─────────────────────────────────────────────────── */
        /* ── 4. draw ─────────────────────────────────────────────────── */
        const uint32_t ic = std::max(cmd.meshInstance->instanceCount, 1u);

        if (cmd.meshInstance->indexBuffer) {
            vkCmdDrawIndexed(cb,
                             cmd.meshInstance->indexCount,
                             ic, // ⚑ instanceCount
                             0, 0, 0);
        } else {
            vkCmdDraw(cb,
                      cmd.meshInstance->vertexCount,
                      ic, // ⚑ instanceCount
                      0, 0);
        }
    }


    bool SceneRenderer::isEntityTreeVisible(Entity e) const {
        // Initialize a flag to determine if we should skip this entity
        bool skipEntity = false;

        // 1) self -----------------------------------------------------------
        if (e.hasComponent<VisibleComponent>() &&
            !e.getComponent<VisibleComponent>().visible)
            return false;

        // 2) ancestors ------------------------------------------------------
        Entity cur = e;
        while (cur.getParent())
        {
            cur = cur.getParent();
            if (cur.hasComponent<VisibleComponent>() &&
                !cur.getComponent<VisibleComponent>().visible)
                return false;
        }
        return true;
    }

    PipelineKey SceneRenderer::makeKey(MeshComponent &mc,
                                       const MeshInstance &mi,
                                       MaterialInstance *mat, Entity &entity) {
        PipelineKey k{};

        /* fixed-function  */
        k.topology = mi.topology;
        k.polygonMode = mc.polygonMode();

        /* shared ids  */
        k.meshId = Utils::crc32(mc.getCacheIdentifier());
        if (entity.hasComponent<MaterialComponent>()) {
            k.vsCRC = Utils::crc32(entity.getComponent<MaterialComponent>().vertexShaderName.string());
            k.fsCRC = Utils::crc32(entity.getComponent<MaterialComponent>().fragmentShaderName.string());
        } else {
            k.vsCRC = Utils::crc32("DefaultShaderKey.vert");
            k.fsCRC = Utils::crc32("DefaultShaderKey.frag");
        }

        k.materialFlags = mat && mat->baseColorTexture ? 1u : 0u;
        return k;
    }

    PipelineInfo SceneRenderer::makePipelineInfo(MaterialInstance *mat) {
        PipelineInfo info{};
        /* descriptor set layouts */
        info.setLayouts.resize(mat ? 2 : 1);

        info.setLayouts[0] = descriptorRegistry.getManager(DescriptorManagerType::MVP)
                .getDescriptorSetLayout();
        info.setLayouts[1] = mat
                                 ? descriptorRegistry.getManager(DescriptorManagerType::MaterialData)
                                 .getDescriptorSetLayout()
                                 : VK_NULL_HANDLE;

        /* vertex input: binding 0 (mesh), binding 1 (instance) */
        info.bindings = {
            {
                {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX},
            }
        };

        info.attrs = {
            {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3},
                {2, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 6},
                {3, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 8},
                {4, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 10},
            }
        };
        info.attrCount = 5;

        info.materialInstance = mat;

        return info;
    }

    std::shared_ptr<MaterialInstance>
    SceneRenderer::getMaterialInstance(Entity entity) {
        if (!entity.hasComponent<MaterialComponent>()) return nullptr;

        auto &matC = entity.getComponent<MaterialComponent>();
        auto it = m_materialInstances.find(entity.getUUID());
        if (it != m_materialInstances.end()) return it->second;

        auto mi = initializeMaterial(entity, matC);
        m_materialInstances[entity.getUUID()] = mi;
        return mi;
    }

    std::shared_ptr<MaterialInstance> SceneRenderer::initializeMaterial(
        Entity entity, const MaterialComponent &materialComponent) {
        auto materialInstance = std::make_shared<MaterialInstance>();

        auto texAsset = assetManager()->get<TextureAsset>(materialComponent.albedoTexturePath.string());

        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageCI.extent.width = texAsset->width;
        imageCI.extent.height = texAsset->height;
        imageCI.extent.depth = texAsset->depth;

        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        VulkanImageCreateInfo vulkanImageCreateInfo(m_context->vkDevice(), m_context->allocator(), imageCI,
                                                    imageViewCI);
        vulkanImageCreateInfo.setLayout = true;
        vulkanImageCreateInfo.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkanImageCreateInfo.dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vulkanImageCreateInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vulkanImageCreateInfo.debugInfo = materialComponent.albedoTexturePath.string();
        vulkanImageCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        auto vulkanImage = cache()->images.get(vulkanImageCreateInfo);

        VulkanTexture2DCreateInfo texCreateInfo{ m_context->vkDevice(), texAsset};
        texCreateInfo.image = vulkanImage;

        auto vkTex = cache()->textures.get(texCreateInfo);
        materialInstance->baseColorTexture = vkTex;

        // 1 Load Shader code
        auto vsSPV = assetManager()->get<SPIRVAsset>(materialComponent.vertexShaderName.string());
        auto fsSPV = assetManager()->get<SPIRVAsset>(materialComponent.fragmentShaderName.string());
        // 2) Wrap into a GPU resource
        VulkanShaderModuleCreateInfo vertexShaderCreateInfo(m_context->vkDevice(), vsSPV, VK_SHADER_STAGE_VERTEX_BIT,
                                                            materialComponent.vertexShaderName.string());
        VulkanShaderModuleCreateInfo fragmentShaderCreateInfo(m_context->vkDevice(), fsSPV,
                                                              VK_SHADER_STAGE_FRAGMENT_BIT,
                                                              materialComponent.fragmentShaderName.string());
        // 3) Later in pipeline creation:

        // 3) Ask the GPU cache for shared modules:
        auto vsModule = cache()->shaderModules.get(vertexShaderCreateInfo);
        auto fsModule = cache()->shaderModules.get(fragmentShaderCreateInfo);

        materialInstance->addShader(vsModule);
        materialInstance->addShader(fsModule);

        Log::Logger::getInstance()->info("Created Material for Entity: {}", entity.getName());
        return materialInstance;
    }

    void SceneRenderer::debugPrintStats() const {
        Log::Logger::getInstance()->trace(
            "Frame: ent={}  batches={}  instances={}  draws={}  saved={}"
            " | CPU collect {:.2f} ms  record {:.2f} ms | GPU {:.2f} ms",
            m_stats.entityCount,
            m_stats.batchCount,
            m_stats.instanceTotal,
            m_stats.drawCalls,
            m_stats.drawCallsSaved(),
            m_stats.asMs(m_stats.cpuCollectNs),
            m_stats.asMs(m_stats.cpuRecordNs),
            m_stats.asMs(m_stats.gpuNs));
    }

    std::unordered_map<DescriptorManagerType, VkDescriptorSet> SceneRenderer::buildMaterialSamplerSet(
        std::shared_ptr<MaterialInstance> &matInst) {
        std::unordered_map<DescriptorManagerType, VkDescriptorSet> out;

        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &matInst->baseColorTexture->getDescriptorInfo();
        out[DescriptorManagerType::MaterialSampler]
                = descriptorRegistry
                .getManager(DescriptorManagerType::MaterialSampler)
                .getOrCreateDescriptorSet({w});

        return out;
    }
}
