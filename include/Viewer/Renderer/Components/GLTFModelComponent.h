//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_GLTFMODELCOMPONENT_H
#define MULTISENSE_VIEWER_GLTFMODELCOMPONENT_H

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/detail/type_quat.hpp>
#include <glm/fwd.hpp>
#include <vulkan/vulkan_core.h>
#include <cfloat>

#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Texture.h"
#include "Viewer/Renderer/Components/GLTFDefines.h"


namespace VkRender {
    struct GLTFModelComponent {
        std::unique_ptr<Model> model;
        std::string memberVariable = "Hello";

        GLTFModelComponent() = default;

        GLTFModelComponent(const GLTFModelComponent &) = default;

        GLTFModelComponent &operator=(const GLTFModelComponent &other) {
            return *this;
        }

        GLTFModelComponent(const std::filesystem::path &modelPath, VulkanDevice *device) {
            model = std::make_unique<Model>(modelPath, device);
        }


    };
};


#endif //MULTISENSE_VIEWER_GLTFMODELCOMPONENT_H
