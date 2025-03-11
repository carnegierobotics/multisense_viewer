//
// Created by magnus on 3/11/25.
//

#ifndef UBUNTUKEYINPUT_H
#define UBUNTUKEYINPUT_H

#include "Viewer/Rendering/Core/KeyInput.h"


namespace VkRender {
    class UbuntuKeyInput : public Input {
    public:

    protected:
        bool isKeyPressedImpl(int keyCode) override;
    };
}




#endif //UBUNTUKEYINPUT_H
