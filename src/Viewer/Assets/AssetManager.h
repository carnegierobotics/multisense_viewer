//
// Created by mgjer on 27/04/2025.
//

#ifndef ASSETMANAGER_H
#define ASSETMANAGER_H

#include <filesystem>

#include "Viewer/Assets/IAssetLoader.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <typeindex>
#include <type_traits>

namespace VkRender {
    class AssetManager {
    public:
        void registerLoader(std::unique_ptr<IAssetLoader> loader) {
            m_loaders.emplace_back(std::move(loader));
        }

        template<typename T>
        std::shared_ptr<T> get(const std::filesystem::path &filepath) {
            // what we ultimately want
            const std::type_index wanted = typeid(T);
            // STEP 1: collect every loader that says it CAN load this file
            std::vector<IAssetLoader *> candidates;
            for (auto &loader: m_loaders) {
                if (loader->canLoad(filepath.string()))
                    candidates.push_back(loader.get());
            }

            // no one claims to handle ".ply" (or whatever)
            if (candidates.empty()) {
                throw std::runtime_error(
                    "ASSETMANAGER: No loader registered for “" + filepath.string() + "”");
            }

            // STEP 2: among those candidates, find the one that produces T
            for (auto *loader: candidates) {
                if (loader->assetType() == wanted) {
                    // STEP 3: bingo—load and cast
                    return std::static_pointer_cast<T>(loader->load(filepath));
                }
            }

            // we found other loaders (e.g. QuadricLoader), but none for Gaussian2DAsset
            std::ostringstream oss;
            oss << "ASSETMANAGER: File “" << filepath.string() << "” is loadable by:";
            for (auto *loader: candidates)
                oss << "\n  - [" << loader->assetType().name() << "]";

            oss << "\nBut you requested: [" << wanted.name() << "]";
            throw std::runtime_error(oss.str());
        }

    private:
        std::vector<std::unique_ptr<IAssetLoader> > m_loaders;
        std::unordered_map<std::string, std::shared_ptr<BaseAsset> > m_cache;
        std::mutex m_mutex; // TODO Async testing
    };
}
#endif //ASSETMANAGER_H
