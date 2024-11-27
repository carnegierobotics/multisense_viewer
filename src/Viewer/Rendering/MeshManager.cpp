//
// Created by magnus on 11/27/24.
//

#include "Viewer/Rendering/MeshManager.h"

namespace VkRender{


    std::shared_ptr<MeshData> MeshManager::getMeshData(MeshComponent& meshComponent) {
        std::lock_guard<std::mutex> lock(cacheMutex);

        std::string identifier = meshComponent.getCacheIdentifier();

        auto it = meshDataCache.find(identifier);
        if (it != meshDataCache.end()) {
            return it->second;
        }

        auto meshData = loadMeshData(meshComponent.meshDataType(), meshComponent.modelPath());
        if (meshData) {
            meshDataCache[identifier] = meshData;
        }

        return meshData;
    }

    void MeshManager::clearCache() {

    }

    void MeshManager::removeMeshData(const std::string &identifier) {

    }

    std::shared_ptr<MeshData> MeshManager::loadMeshData(MeshDataType type, const std::filesystem::path& path) {
        return std::make_shared<MeshData>(type, path);

    }
}