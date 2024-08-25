//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_PROPERTIESLAYER
#define MULTISENSE_VIEWER_PROPERTIESLAYER

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/ImGui/LayerUtils.h"
#include <glm/gtc/type_ptr.hpp>  // Include this header for glm::value_ptr

namespace VkRender {


    class PropertiesLayer : public Layer {
    public:
        void onAttach() override;

        void onDetach() override;

        void onUIRender(GuiObjectHandles &handles) override;

        void onFinishedRender() override;

        void setSelectedEntity(Entity entity);

    public:
        template<typename T, typename UIFunction>
        void drawComponent(const std::string &name, Entity entity, UIFunction uiFunction);

        void drawComponents(VkRender::GuiObjectHandles &handles, Entity entity);

        Entity m_selectionContext;
        std::future<LayerUtils::LoadFileInfo> loadFileFuture;

        template<typename T>
        void displayAddComponentEntry(const std::string &entryName);

        void checkFileImportCompletion(GuiObjectHandles &handles);

        void handleSelectedFile(const LayerUtils::LoadFileInfo &loadFileInfo, GuiObjectHandles &handles);

        void openImportFileDialog(const std::string &fileDescription, const std::vector<std::string> &type,
                                  LayerUtils::FileTypeLoadFlow flow);

        static void
        drawVec3Control(const std::string &label, glm::vec3 &values, float resetValue, float columnWidth, float speed);
    };
}

#endif //MULTISENSE_VIEWER_PROPERTIESLAYER
