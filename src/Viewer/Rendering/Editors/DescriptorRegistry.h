//
// Created by magnus-desktop on 11/17/24.
//

#ifndef DESCRIPTORMANAGERREGISTRY_H
#define DESCRIPTORMANAGERREGISTRY_H

#include "Viewer/Rendering/Editors/DescriptorSetManager.h"


namespace VkRender {
    enum class DescriptorType {
        MVP,
        Material,
        DynamicCameraGizmo,
        Viewport3DTexture,
    };

    class DescriptorRegistry {
    public:
        DescriptorSetManager &getManager(DescriptorType type) {
            if (!m_managers[static_cast<size_t>(type)]) {
                throw std::runtime_error("Tried to get DescriptorManager that was not initialized");
            }
            return *m_managers[static_cast<size_t>(type)];
        }

        void createManager(
                DescriptorType type,
                VulkanDevice &device
        ) {
            std::vector<VkDescriptorSetLayoutBinding> bindings;


            switch (type) {
                case DescriptorType::MVP:
                    bindings = {
                            {
                                    0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                                                             VK_SHADER_STAGE_VERTEX_BIT |
                                                                             VK_SHADER_STAGE_FRAGMENT_BIT,
                                    nullptr
                            },
                            {
                                    1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,
                                    nullptr
                            },
                    };

                    break;
                case DescriptorType::Material:
                    bindings = {
                            {
                                    0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                    nullptr
                            },
                            {
                                    1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                    nullptr
                            },
                    };
                    break;
                case DescriptorType::DynamicCameraGizmo:
                    bindings = {
                            {
                                    0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,
                                    nullptr
                            }
                    };
                    break;
                case DescriptorType::Viewport3DTexture:
                    bindings = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                                {
                                 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT |
                                                                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                                                                                                                nullptr
                                }};
                    break;
            }


            m_managers[static_cast<size_t>(type)] =
                    std::make_unique<DescriptorSetManager>(device, bindings);
        }

    private:
        std::array<std::unique_ptr<DescriptorSetManager>, 4> m_managers;
    };

}
#endif //DESCRIPTORMANAGERREGISTRY_H
