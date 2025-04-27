//
// Created by mgjer on 27/04/2025.
//

#ifndef ASSETMANAGER_H
#define ASSETMANAGER_H

#include "IAssetLoader.h"
#include <memory>
#include <unordered_map>
#include <mutex>
#include <type_traits>

namespace VkRender {


    class AssetManager {
    public:
        void registerLoader(std::unique_ptr<IAssetLoader> loader) {
            m_loaders.emplace_back(std::move(loader));
        }

        template<typename T>
     std::shared_ptr<T> get(const std::string &key)
        {
            static_assert(std::is_base_of<BaseAsset, T>::value,
                          "T must derive from BaseAsset");

            // 1) Check cache
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = m_cache.find(key);
                if (it != m_cache.end())
                    return std::static_pointer_cast<T>(it->second);
            }

            // 2) Find a loader that can handle this key
            for (auto& loader : m_loaders) {
                if (loader->canLoad(key)) {
                    auto asset = loader->load(key);
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        m_cache[key] = asset;
                    }
                    return std::static_pointer_cast<T>(asset);
                }
            }

            throw std::runtime_error("AssetManager: no loader for " + key);
        }

    private:
        std::vector<std::unique_ptr<IAssetLoader>> m_loaders;
        std::unordered_map<std::string, std::shared_ptr<BaseAsset>> m_cache;
        std::mutex m_mutex; // TODO Async testing
    };


}
#endif //ASSETMANAGER_H
