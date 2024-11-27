//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_MESHMANAGER_H
#define MULTISENSE_VIEWER_MESHMANAGER_H

#include "Viewer/Rendering/MeshData.h"
#include "Viewer/Rendering/Components/MeshComponent.h"

namespace VkRender{
    class MeshManager {
    public:
        MeshManager() = default;

        std::shared_ptr<MeshData> getMeshData(MeshComponent& meshComponent);

        void clearCache();
        void removeMeshData(const std::string& identifier);

    private:
        std::unordered_map<std::string, std::shared_ptr<MeshData>> meshDataCache;
        std::mutex cacheMutex;

        std::shared_ptr<MeshData> loadMeshData(MeshDataType type, const std::filesystem::path& path);
    };

}

#endif //MULTISENSE_VIEWER_MESHMANAGER_H
