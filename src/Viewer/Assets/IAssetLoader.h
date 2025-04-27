//
// Created by mgjer on 27/04/2025.
//

#ifndef IASSETLOADER_H
#define IASSETLOADER_H

#include <memory>
#include <string>

namespace VkRender {

struct BaseAsset { virtual ~BaseAsset() = default; };

struct IAssetLoader {
    virtual ~IAssetLoader() = default;

    // Can this loader handle this file?
    virtual bool canLoad(const std::string& key) const = 0;

    // Do the actual load; the returned BaseAsset must be castable to the expected type.
    virtual std::shared_ptr<BaseAsset> load(const std::string& key) = 0;
};

}

#endif //IASSETLOADER_H
