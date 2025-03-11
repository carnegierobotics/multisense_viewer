//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_MESHMANAGER_H
#define MULTISENSE_VIEWER_MESHMANAGER_H

#include "Viewer/Rendering/MeshData.h"
#include "Viewer/Rendering/Components/MeshComponent.h"

namespace VkRender {
    class MeshManager {
    public:
        static MeshManager& instance() {
            static MeshManager instance;
            return instance;
        }

        std::shared_ptr<MeshData> getMeshData(MeshComponent& meshComponent);
        void clearCache();
        void removeMeshData(const std::string& identifier);

    private:
        MeshManager() = default;
        std::unordered_map<std::string, std::shared_ptr<MeshData>> meshDataCache;
        std::mutex cacheMutex;
        std::shared_ptr<MeshData> loadMeshData(MeshComponent& meshComponent);

        // Delete copy/move to enforce singleton semantics
        MeshManager(const MeshManager&) = delete;
        MeshManager& operator=(const MeshManager&) = delete;
    };
}

#endif //MULTISENSE_VIEWER_MESHMANAGER_H
