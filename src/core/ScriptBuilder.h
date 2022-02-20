#pragma once

//
// Created by magnus on 9/6/21.
//

#ifndef AR_ENGINE_SCRIPTBUILDER_H
#define AR_ENGINE_SCRIPTBUILDER_H
#include <memory>
#include <map>
#include <iostream>
#include <vector>
#include <cassert>
#include "VulkanDevice.h"
#include "Base.h"

// Based of self registering factory
// cppstories https://www.cppstories.com/2018/02/factory-selfregister/


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

    static std::unique_ptr<Base> Create(const std::string& name) {
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




#endif //AR_ENGINE_SCRIPTBUILDER_H