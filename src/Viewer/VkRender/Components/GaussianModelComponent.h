//
// Created by magnus on 8/15/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANMODELCOMPONENT_H
#define MULTISENSE_VIEWER_GAUSSIANMODELCOMPONENT_H

#include "Viewer/VkRender/pch.h"
#include "MeshComponent.h"

namespace VkRender {
    class GaussianModelComponent {
    public:

        GaussianModelComponent() = delete;

        GaussianModelComponent(const GaussianModelComponent &) = delete;

        GaussianModelComponent &operator=(const GaussianModelComponent &other) {
            return *this;
        }

        explicit GaussianModelComponent(std::filesystem::path filePath);


        struct GaussianPoints {
            std::vector<glm::vec3> positions;
            std::vector<glm::quat> quats;
            std::vector<glm::vec3> scales;
            std::vector<float> opacities;
            std::vector<float> sphericalHarmonics;  // Add this line
            uint32_t shDim = 1; // default rgb
            [[nodiscard]] uint32_t getSize() const {
                return positions.size();
            }

            uint32_t getShDim() const {
                return shDim; // TODO implement
            }
        };


        const GaussianPoints& getGaussians() const{
            return m_gaussians;
        }

        MeshComponent* getMeshComponent() {return m_imageComponent.get();}

    private:
        GaussianPoints m_gaussians;
        std::unique_ptr<MeshComponent> m_imageComponent;

    private:
        GaussianPoints loadFromFile(std::filesystem::path path, int downSampleRate);

    };


}

#endif //MULTISENSE_VIEWER_GAUSSIANMODELCOMPONENT_H
