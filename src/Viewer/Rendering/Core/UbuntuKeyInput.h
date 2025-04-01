//
// Created by magnus on 3/11/25.
//

#ifndef UBUNTUKEYINPUT_H
#define UBUNTUKEYINPUT_H

#include <unordered_map>

#include "Viewer/Rendering/Core/KeyInput.h"


namespace VkRender {
    class UbuntuKeyInput : public Input {

    std::unordered_map<int, bool> previousKeyStates;

        bool isKeyPressedImpl(int keyCode) override;
        bool isKeyClickedImpl(int keyCode) override;
    };
}




#endif //UBUNTUKEYINPUT_H
