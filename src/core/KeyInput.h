//
// Created by magnus on 9/14/22.
//

#ifndef MULTISENSE_VIEWER_KEYINPUT_H
#define MULTISENSE_VIEWER_KEYINPUT_H

#include "GLFW/glfw3.h"

struct Input {

    [[nodiscard]] bool getButtonDown(int key) const {
        if (key == *lastKeyPress && *action == GLFW_PRESS)
            return true;

        return false;
    }

    [[nodiscard]] bool getButtonUp(int key) const {
        if (key == *lastKeyPress && *action == GLFW_RELEASE)
            return true;

        return false;
    }

    [[nodiscard]] bool getButton(int key) const {
        if (key == *lastKeyPress)
            return true;

        return false;
    }

    const int* action;
    const int* lastKeyPress;

};


#endif //MULTISENSE_VIEWER_KEYINPUT_H
