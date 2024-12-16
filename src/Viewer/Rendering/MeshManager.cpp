//
// Created by magnus on 11/27/24.
//

#include "Viewer/Rendering/MeshManager.h"

namespace VkRender{


    std::shared_ptr<MeshData> MeshManager::getMeshData(MeshComponent& meshComponent) {
        std::lock_guard<std::mutex> lock(cacheMutex);
        std::string identifier = meshComponent.getCacheIdentifier();
        auto it = meshDataCache.find(identifier);
        if (it != meshDataCache.end() && !meshComponent.updateMeshData) {
            return it->second;
        }
        if (meshComponent.data()) {
            auto meshData = meshComponent.data()->generateMeshData();
            meshDataCache[identifier] = meshData;
            meshData->isDirty = true;
            return meshData;
        }
        return nullptr;
    }
    void MeshManager::clearCache() {

    }

    void MeshManager::removeMeshData(const std::string &identifier) {

    }
}