//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_LAYER_H
#define MULTISENSE_LAYER_H


namespace ArEngine {
    struct GuiLayerUpdateInfo {
        bool firstFrame;
        float frameTimer;
        float width;
        float height;
        std::string deviceName;
        std::string title;

        std::array<float, 50> frameTimes{};
        float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;

    };

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual void OnAttach() {}

        virtual void OnDetach() {}

        virtual void OnUIRender() {}

        void updateInfo(GuiLayerUpdateInfo *_info) {
            this->info = _info;
        }


    protected:
        GuiLayerUpdateInfo *info;
    };


};


#endif //MULTISENSE_LAYER_H
