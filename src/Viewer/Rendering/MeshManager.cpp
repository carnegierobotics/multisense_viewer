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
            if (it->second->isDirty) {
                auto meshData = meshComponent.data()->generateMeshData();
                if (meshData) {
                    it->second = meshData;
                    it->second->isDirty = false;
                }
                Log::Logger::getInstance()->warning("Failed to retrive/generate Mesh Data");
            }

            return it->second;
        }
        if (meshComponent.data()) {
            Utils::ScopedTimer timer("Generating Material Instance for mesh: " + identifier);
            auto meshData = meshComponent.data()->generateMeshData();
            meshDataCache[identifier] = meshData;
            return meshData;
        }
        return nullptr;
    }
    void MeshManager::clearCache() {

    }

    void MeshManager::removeMeshData(const std::string &identifier) {

    }
}