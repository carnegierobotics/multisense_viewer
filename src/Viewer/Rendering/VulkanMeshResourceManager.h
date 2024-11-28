//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_VULKANMESHRESOURCEMANAGER_H
#define MULTISENSE_VIEWER_VULKANMESHRESOURCEMANAGER_H


#include "Viewer/Rendering/MeshData.h"
#include "Viewer/Rendering/MeshInstance.h"

namespace VkRender {
    class Application;

    class MeshResourceManager {
    public:
        explicit MeshResourceManager(Application* context)
                : m_context(context) {}

        void
        updateMeshInstance(const std::string &identifier, const std::shared_ptr<MeshData> &meshData);

        std::shared_ptr<MeshInstance>
        getMeshInstance(const std::string &identifier, const std::shared_ptr<MeshData> &meshData,
                        MeshDataType meshType);

        void clearCache();
        void removeMeshInstance(const std::string& identifier);

    private:
        Application* m_context{};
        std::unordered_map<std::string, std::shared_ptr<MeshInstance>> meshInstanceCache{};
        std::mutex cacheMutex{};


        std::shared_ptr<MeshInstance>
        createMeshInstance(const std::shared_ptr<MeshData> &meshData, MeshDataType meshType);


    };
}

#endif //MULTISENSE_VIEWER_VULKANMESHRESOURCEMANAGER_H
