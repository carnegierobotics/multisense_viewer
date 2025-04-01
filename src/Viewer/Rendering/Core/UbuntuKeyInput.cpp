//
// Created by magnus on 3/11/25.
//

#include "UbuntuKeyInput.h"
#include "Viewer/Application/Application.h"

#include <GLFW/glfw3.h>


namespace VkRender {

    Input* Input::s_instance = new UbuntuKeyInput();


    bool UbuntuKeyInput::isKeyPressedImpl(int keyCode) {
        auto glfwWindow = Application::instance().getWindow();
        auto state = glfwGetKey(glfwWindow, keyCode);
        return state == GLFW_PRESS || state == GLFW_REPEAT;
    }


    bool UbuntuKeyInput::isKeyClickedImpl(int keyCode) {
        auto glfwWindow = Application::instance().getWindow();
        auto currentState = glfwGetKey(glfwWindow, keyCode) == GLFW_PRESS;

        bool wasPressedBefore = previousKeyStates[keyCode];
        previousKeyStates[keyCode] = currentState;

        return currentState && !wasPressedBefore;
    }


}
