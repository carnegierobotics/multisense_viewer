//
// Created by magnus on 11/27/24.
//

#include "Viewer/Rendering/MeshManager.h"

namespace VkRender{


    std::shared_ptr<MeshData> MeshManager::getMeshData(MeshComponent& meshComponent) {
        std::lock_guard<std::mutex> lock(cacheMutex);
        std::string identifier = meshComponent.getCacheIdentifier();
        auto it = meshDataCache.find(identifier);
        if (it != meshDataCache.end() && !meshComponent.isDirty) {
            return it->second;
        }
        if (meshComponent.data()) {
            auto meshData = meshComponent.data()->generateMeshData();
            meshDataCache[identifier] = meshData;
            meshComponent.isDirty = false;
            return meshData;
        }
        return nullptr;
    }
    void MeshManager::clearCache() {

    }

    void MeshManager::removeMeshData(const std::string &identifier) {

    }
}