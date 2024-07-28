/**
 * @file: MultiSense-Viewer/include/Viewer/VkRender/Core/KeyInput.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-09-14, mgjerde@carnegierobotics.com, Created file.
 **/


#ifndef MULTISENSE_VIEWER_KEYINPUT_H
#define MULTISENSE_VIEWER_KEYINPUT_H


#ifdef APIENTRY
#undef APIENTRY
#endif
#include <GLFW/glfw3.h>

struct Input {

    Input(){
        action = 0;
        lastKeyPress = 0;
    }

    [[nodiscard]] bool getButtonDown(int key) const {
        if (key == lastKeyPress && action == GLFW_PRESS)
            return true;

        return false;
    }

    /**@brief Not currently implemented */
    [[nodiscard]] bool getButtonUp(int key) const {
        if (key == lastKeyPress && action == GLFW_RELEASE)
            return true;

        return false;
    }

    [[nodiscard]] bool getButton(int key) const {
        if (key == lastKeyPress)
            return true;

        return false;
    }

    int action;
    int lastKeyPress;

    struct keys{
        bool up = false;
        bool down = false;
        bool left = false;
        bool right = false;
    }keys;

};


#endif //MULTISENSE_VIEWER_KEYINPUT_H