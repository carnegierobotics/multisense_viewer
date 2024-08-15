//
// Created by magnus on 8/15/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANMODELGRAPHICSPIPELINE_H
#define MULTISENSE_VIEWER_GAUSSIANMODELGRAPHICSPIPELINE_H


#include "Viewer/VkRender/Core/CommandBuffer.h"
#ifdef SYCL_ENABLED
#include <sycl/sycl.hpp>

#endif
namespace VkRender {
#ifdef SYCL_ENABLED
    class GaussianModelGraphicsPipeline {
    public:
        GaussianModelGraphicsPipeline();

        template<typename T>
        void bind(T &modelComponent);

        void draw(CommandBuffer &cmdBuffers);

    private:

        sycl::queue queue{};


    };
#else
    class GaussianModelGraphicsPipeline {
    public:
        void draw(CommandBuffer&) { /* no-op */ }
        template <typename T>
        void bind(T&) { /* no-op */ }
    };
#endif
}

#endif //MULTISENSE_VIEWER_GAUSSIANMODELGRAPHICSPIPELINE_H
