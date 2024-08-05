//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_SCENE_H
#define MULTISENSE_VIEWER_SCENE_H


namespace VkRender {
    class Renderer;

    class Scene {

    public:
        Scene() = delete;
        explicit Scene(Renderer& ctx) : m_context(ctx) {}

        virtual void render(CommandBuffer& drawCmdBuffers) = 0;
        virtual void update() = 0;

        // Virtual destructor
        virtual ~Scene() = default;

    protected:
        Renderer& m_context;
    };

}

#endif //MULTISENSE_VIEWER_SCENE_H
