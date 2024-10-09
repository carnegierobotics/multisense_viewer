//
// Created by mgjer on 04/10/2024.
//

#ifndef MATERIALCOMPONENT_H
#define MATERIALCOMPONENT_H

namespace VkRender {
    enum class RenderMode;

    enum class AlphaMode {
        Opaque,
        Mask,
        Blend
    };

    struct MaterialComponent {
        RenderMode renderMode;
        std::string shaderName;
        // Other material properties...

        // PBR Material Properties
        glm::vec4 baseColorFactor = glm::vec4(1.0f); // Default to white
        float metallicFactor = 1.0f;                 // Default metallic
        float roughnessFactor = 1.0f;                // Default rough

        // Textures (may be null if not present)
        std::shared_ptr<Texture> baseColorTexture;
        std::shared_ptr<Texture> metallicRoughnessTexture;
        std::shared_ptr<Texture> normalTexture;
        std::shared_ptr<Texture> occlusionTexture;
        std::shared_ptr<Texture> emissiveTexture;

        // Emissive properties
        glm::vec3 emissiveFactor = glm::vec3(0.0f); // Default to no emission

        // Rendering properties
        AlphaMode alphaMode = AlphaMode::Opaque;
        float alphaCutoff = 0.5f;  // Used if alphaMode is Mask
        bool doubleSided = false;

        // Descriptor set layout information
        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorSet descriptorSet;

        bool usesTexture = false;
        bool usesUniformBuffer = false;

        void initialize(VkDevice device, VkDescriptorPool descriptorPool) {
 // Create descriptor set layout bindings
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    // Binding 0: Material UBO
    VkDescriptorSetLayoutBinding uboBinding = {};
    uboBinding.binding = 0;
    uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings.push_back(uboBinding);

    uint32_t bindingIndex = 1; // Start from 1 for textures

    // Binding for base color texture
    if (baseColorTexture) {
        VkDescriptorSetLayoutBinding textureBinding = {};
        textureBinding.binding = bindingIndex++;
        textureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        textureBinding.descriptorCount = 1;
        textureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(textureBinding);
    }

    // Repeat for other textures (metallicRoughnessTexture, normalTexture, etc.)

    // Create the descriptor set layout
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }

    // Update descriptor set with buffers and images
    std::vector<VkWriteDescriptorSet> descriptorWrites;

    // Material UBO
    VkDescriptorBufferInfo bufferInfo = {};
    //bufferInfo.buffer = materialUBO; // The buffer containing material properties
    bufferInfo.offset = 0;
    //bufferInfo.range = sizeof(MaterialUBOData); // Define a struct matching the UBO in shaders

    VkWriteDescriptorSet uboWrite = {};
    uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    uboWrite.dstSet = descriptorSet;
    uboWrite.dstBinding = 0;
    uboWrite.dstArrayElement = 0;
    uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboWrite.descriptorCount = 1;
    uboWrite.pBufferInfo = &bufferInfo;
    descriptorWrites.push_back(uboWrite);

    // Textures
    bindingIndex = 1;
    if (baseColorTexture) {
        VkDescriptorImageInfo imageInfo = {};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = baseColorTexture->m_view;
        imageInfo.sampler = baseColorTexture->m_sampler;

        VkWriteDescriptorSet textureWrite = {};
        textureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        textureWrite.dstSet = descriptorSet;
        textureWrite.dstBinding = bindingIndex++;
        textureWrite.dstArrayElement = 0;
        textureWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        textureWrite.descriptorCount = 1;
        textureWrite.pImageInfo = &imageInfo;
        descriptorWrites.push_back(textureWrite);
    }

    // Repeat for other textures...

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }

    };

}

#endif //MATERIALCOMPONENT_H
