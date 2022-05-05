//
// Created by magnus on 9/23/21.
//

#ifndef MULTISENSE_UISETTINGS_H
#define MULTISENSE_UISETTINGS_H


#include <utility>
#include <vector>
#include <string>
#include <array>
#include "imgui.h"
#include <memory>


struct UISettings {

public:

    float movementSpeed = 0.2;

    std::array<float, 50> frameTimes{};
    float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;


    bool closeModalPopup = false;

    /** void* for shared data among scripts. User defined type */
    void *physicalCamera = nullptr;
    void *virtualCamera = nullptr;

};


#endif //MULTISENSE_UISETTINGS_H
