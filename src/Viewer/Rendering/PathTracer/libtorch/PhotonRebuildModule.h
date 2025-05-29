//
// Created by magnus on 1/24/25.
//

#ifndef MULTISENSE_VIEWER_PHOTONREBUILDMODULE_H
#define MULTISENSE_VIEWER_PHOTONREBUILDMODULE_H

#include <torch/torch.h>

#include <Viewer/Scenes/Scene.h>

#include "Viewer/Rendering/PathTracer/libtorch/PhotonRebuildFunction.h"

// Wrap your raytracer in a Torch module
namespace VkRender::PathTracer {


    class PhotonRebuildModule : public torch::nn::Module {
    public:
        PhotonRebuildModule(std::weak_ptr<Scene> scene);
        ~PhotonRebuildModule();


        // forward() will trigger a ray trace and return an image tensor
        torch::Tensor forward();

        float* getRenderedImage();
        void freeData();

        void uploadPathTracerFromTensor();
        void uploadSceneFromTensor(std::shared_ptr<Scene> scene);


    private:
        void uploadTensorFromScene(std::weak_ptr<Scene> scene);
        torch::Tensor m_outputTensor;  // Store the output tenso
    };

}


#endif //MULTISENSE_VIEWER_PHOTONREBUILDMODULE_H
