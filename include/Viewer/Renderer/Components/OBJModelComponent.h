//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_OBJMODELCOMPONENT
#define MULTISENSE_VIEWER_OBJMODELCOMPONENT

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/detail/type_quat.hpp>
#include <glm/fwd.hpp>
#include <vulkan/vulkan_core.h>
#include <cfloat>

#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Texture.h"

namespace VkRender {
    struct OBJModelComponent {

        OBJModelComponent() = default;

        OBJModelComponent(const OBJModelComponent &) = default;

        OBJModelComponent &operator=(const OBJModelComponent &other) {
            return *this;
        }

        OBJModelComponent(const std::filesystem::path &modelPath, VulkanDevice *device) {

        }


    };
};


#endif //MULTISENSE_VIEWER_OBJMODELCOMPONENT
