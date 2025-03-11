//
// Created by magnus on 3/10/25.
//

#ifndef SCRIPTABLECOMPONENT_H
#define SCRIPTABLECOMPONENT_H
#include <string_view>
#include <string>

#include "Viewer/Scenes/ScriptableEntity.h"

namespace VkRender {



    // A constexpr function to extract the type name from the function signature
    template <typename T>
    constexpr std::string_view getTypeName() {
#ifdef __clang__
        std::string_view p = __PRETTY_FUNCTION__;
        auto start = p.find("T = ") + 4;
        auto end = p.find(']', start);
        return p.substr(start, end - start);
#elif defined(__GNUC__)
        std::string_view p = __PRETTY_FUNCTION__;
        auto start = p.find("T = ") + 4;
        auto end = p.find(';', start);
        return p.substr(start, end - start);
#elif defined(_MSC_VER)
        std::string_view p = __FUNCSIG__;
        auto start = p.find("getTypeName<") + 12;
        auto end = p.find(">(void)", start);
        return p.substr(start, end - start);
#else
#error Unsupported compiler
#endif
    }


    struct ScriptableComponent {
        ScriptableEntity *instance = nullptr;
        std::string scriptName;  // Added type identifier

        std::function<ScriptableEntity *()> instantiateScript;
        std::function<void()> destroyScript;

        // Overload conversion to bool:
        explicit operator bool() const {
            // Return true if bind() has been called (i.e. instantiateScript is set)
            return static_cast<bool>(instantiateScript);
        }

        template<typename T>
        void bind() {
            scriptName = std::string(getTypeName<T>());

            instantiateScript = [this]() {return reinterpret_cast<ScriptableEntity *>(new T());};
            destroyScript = [this]() {delete  instance; instance = nullptr;};
        }
    };


}


#endif //SCRIPTABLECOMPONENT_H
