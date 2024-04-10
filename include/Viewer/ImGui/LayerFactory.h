//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_LAYERFACTORY_H
#define MULTISENSE_VIEWER_LAYERFACTORY_H

#include "Layer.h"
#include <memory>
#include <string>

class LayerFactory {
public:
    static std::shared_ptr<VkRender::Layer> createLayer(const std::string& layerName);
};


#endif //MULTISENSE_VIEWER_LAYERFACTORY_H
