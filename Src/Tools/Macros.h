//
// Created by magnus on 9/16/21.
//

#ifndef MULTISENSE_MACROS_H
#define MULTISENSE_MACROS_H

#include <vulkan/vulkan_core.h>
#include "string"

#if defined(_MSC_VER)
#define DISABLE_WARNING_PUSH           __pragma(warning( push ))
#define DISABLE_WARNING_POP            __pragma(warning( pop ))
#define DISABLE_WARNING(warningNumber) __pragma(warning( disable : warningNumber ))

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER    DISABLE_WARNING(4100)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION            DISABLE_WARNING(4505)
#define DISABLE_WARNING_EMPTY_BODY
#define DISABLE_WARNING_UNREFERENCED_VARIABLE                    
#define DISABLE_WARNING_UNUSED_VARIABLE                          
#define DISABLE_WARNING_CAST_QUALIFIERS                          
#define DISABLE_WARNING_DOUBLE_PROMOTION                         
#define DISABLE_WARNING_IMPLICIT_FALLTHROUGH                     
#define DISABLE_WARNING_TYPE_LIMITS                              
#define DISABLE_WARNING_MISSING_INITIALIZERS                     
// other warnings you want to deactivate...

#elif defined(__GNUC__) || defined(__clang__)
#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING_PUSH                                      DO_PRAGMA(GCC diagnostic push)
#define DISABLE_WARNING_POP                                       DO_PRAGMA(GCC diagnostic pop)
#define DISABLE_WARNING(warningName)                              DO_PRAGMA(GCC diagnostic ignored #warningName)

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER             DISABLE_WARNING(-Wunused-parameter)
#define DISABLE_WARNING_EMPTY_BODY                                DISABLE_WARNING(-Wempty-body)
#define DISABLE_WARNING_UNREFERENCED_VARIABLE                     DISABLE_WARNING(-Wunused-variable)
#define DISABLE_WARNING_UNUSED_VARIABLE                           DISABLE_WARNING(-Wunused-value)
#define DISABLE_WARNING_CAST_QUALIFIERS                           DISABLE_WARNING(-Wcast-qual)
#define DISABLE_WARNING_DOUBLE_PROMOTION                          DISABLE_WARNING(-Wdouble-promotion)
#define DISABLE_WARNING_IMPLICIT_FALLTHROUGH                      DISABLE_WARNING(-Wimplicit-fallthrough)
#define DISABLE_WARNING_TYPE_LIMITS                               DISABLE_WARNING(-Wtype-limits)
#define DISABLE_WARNING_MISSING_INITIALIZERS                      DISABLE_WARNING(-Wmissing-field-initializers)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION                     DISABLE_WARNING(-Wunused-function)
// other warnings you want to deactivate...

#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
// other warnings you want to deactivate...

#endif

#define CHECK_RESULT(f) \
{                                                                                                           \
    VkResult res = (f);                                                                                     \
    if (res != VK_SUCCESS)                                                                                  \
    {                                                                                                       \
        std::cerr << "Fatal : VkResult is \"" << Macros::errorString(res) << "\" in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        assert(res == VK_SUCCESS);                                                                                \
    }                                                                                                             \
}

namespace Macros {


    inline std::string errorString(VkResult errorCode) {
        switch (errorCode) {
#define STR(r) case VK_ ##r: return #r
            STR(NOT_READY);
            STR(TIMEOUT);
            STR(EVENT_SET);
            STR(EVENT_RESET);
            STR(INCOMPLETE);
            STR(ERROR_OUT_OF_HOST_MEMORY);
            STR(ERROR_OUT_OF_DEVICE_MEMORY);
            STR(ERROR_INITIALIZATION_FAILED);
            STR(ERROR_DEVICE_LOST);
            STR(ERROR_MEMORY_MAP_FAILED);
            STR(ERROR_LAYER_NOT_PRESENT);
            STR(ERROR_EXTENSION_NOT_PRESENT);
            STR(ERROR_FEATURE_NOT_PRESENT);
            STR(ERROR_INCOMPATIBLE_DRIVER);
            STR(ERROR_TOO_MANY_OBJECTS);
            STR(ERROR_FORMAT_NOT_SUPPORTED);
            STR(ERROR_SURFACE_LOST_KHR);
            STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
            STR(SUBOPTIMAL_KHR);
            STR(ERROR_OUT_OF_DATE_KHR);
            STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
            STR(ERROR_VALIDATION_FAILED_EXT);
            STR(ERROR_INVALID_SHADER_NV);
#undef STR
            default:
                return "UNKNOWN_ERROR";
        }
    }
}

#endif //MULTISENSE_MACROS_H
