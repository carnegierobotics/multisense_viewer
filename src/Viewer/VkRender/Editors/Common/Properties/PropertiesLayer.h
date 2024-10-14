//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_PROPERTIESLAYER
#define MULTISENSE_VIEWER_PROPERTIESLAYER

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/ImGui/LayerUtils.h"
#include <glm/gtc/type_ptr.hpp>  // Include this header for glm::value_ptr

namespace VkRender {


    class PropertiesLayer : public Layer {
    public:
        void onAttach() override;

        void onDetach() override;

        void onUIRender() override;

        void onFinishedRender() override;

        void setScene(std::shared_ptr<Scene> scene) override;

        void setSelectedEntity(Entity entity);

    public:
        template<typename T, typename UIFunction>
        void drawComponent(const std::string &name, Entity entity, UIFunction uiFunction);

        void drawComponents(Entity entity);

        Entity m_selectionContext;
        std::future<LayerUtils::LoadFileInfo> loadFileFuture;
        std::future<LayerUtils::LoadFileInfo> loadFolderFuture;
        char m_tagBuffer[256];  // Adjust size as needed
        bool m_needsTagUpdate = true;

        template<typename T>
        void displayAddComponentEntry(const std::string &entryName);

        void handleSelectedFileOrFolder(const LayerUtils::LoadFileInfo &loadFileInfo);

        void checkFileImportCompletion();

        void checkFolderImportCompletion();

        void openImportFileDialog(const std::string &fileDescription, const std::vector<std::string> &type,
                                  LayerUtils::FileTypeLoadFlow flow);

        void openImportFolderDialog(const std::string &fileDescription, const std::vector<std::string> &type,
                                    LayerUtils::FileTypeLoadFlow flow);

        static void
        drawVec3Control(const std::string &label, glm::vec3 &values, float resetValue, float columnWidth, float speed);

        static void drawFloatControl(const std::string &label, float &value, float resetValue, float speed, float columnWidth);
    };
}

#endif //MULTISENSE_VIEWER_PROPERTIESLAYER