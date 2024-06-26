/**
 * @file: MultiSense-Viewer/include/Viewer/Scripts/Private/ScriptBuilder.h
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
 *   2021-9-6, mgjerde@carnegierobotics.com, Created file.
 **/
#pragma once

#ifndef MULTISENSE_SCRIPTBUILDER_H
#define MULTISENSE_SCRIPTBUILDER_H

#include <memory>
#include <map>
#include <iostream>
#include <vector>
#include <cassert>

#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Scripts/Private/Base.h"
#include "Viewer/Core/CommandBuffer.h"

// Based of self registering factory
// cppstories https://www.cppstories.com/2018/02/factory-selfregister/

namespace VkRender {


class ComponentMethodFactory {
public:
    using TCreateMethod = std::unique_ptr<Base>(*)();
    TCreateMethod m_CreateFunc;
    std::string description;

public:
    ComponentMethodFactory() = delete;

    static bool Register(const std::string name, TCreateMethod createFunc) {

        if (auto it = s_methods.find(name); it == s_methods.end()) {
            s_methods[name] = createFunc;
            return true;
        }
        return false;
    }

    static std::shared_ptr<Base> Create(const std::string& name) {
        if (auto it = s_methods.find(name); it != s_methods.end()) {
            return it->second();
        }
        return nullptr;
    }

private:
    static std::map<std::string, TCreateMethod> s_methods;
};

inline std::map<std::string, ComponentMethodFactory::TCreateMethod> ComponentMethodFactory::s_methods;

template<typename T>
class RegisteredInFactory {
protected:
    static bool s_bRegistered;
};

template<typename T >
bool RegisteredInFactory<T>::s_bRegistered = ComponentMethodFactory::Register(T::GetFactoryName(), T::CreateMethod);
}

#endif //MULTISENSE_SCRIPTBUILDER_H