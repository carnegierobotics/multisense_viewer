# CMAKE FILE DESCRIPTION
# - Sets build type to Release if none was specified
# - Pulls submodules into /external folder
# - Adds each submodule to project add_subdirectory/include_directory
# - Adds internal library Autoconnect
# - Option to enabled all warnings when compiling with GCC

set(CRL_SERVER_IP 35.211.65.110:80)
set(CRL_SERVER_PROTOCOL http)
set(CRL_SERVER_DESTINATION /api.php)
set(CRL_SERVER_VERSIONINFO_DESTINATION /version.php)
set(VIEWER_LOG_LEVEL LOG_INFO)

find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" OFF)
    if (GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if (NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif ()
    endif ()
endif ()
# Include the proprietary GigE-Vision Module
if(PROPRIETARY_GIGEVISION_MODULE)
    if(NOT GIGEVISION_MODULE_PATH)
        message(FATAL_ERROR "GIGEVISION_MODULE_PATH is required when PROPRIETARY_GIGEVISION_MODULE is ON")
    endif()
    # Include directories for the proprietary module
    include_directories(${GIGEVISION_MODULE_PATH}/include)

    # Add the proprietary module library
    # Assuming the proprietary module provides a library file, you can link it like this:
    set(PROPRIETARY_MODULE_LIB ${GIGEVISION_MODULE_PATH}/lib/libcrlgev.a)

    # Check if the library exists
    if(NOT EXISTS ${PROPRIETARY_MODULE_LIB})
        message(FATAL_ERROR "Proprietary module library not found at ${PROPRIETARY_MODULE_LIB}")
    endif()

endif()
# Include Submodules into project.
# Check if exists or display fatal error
set(GLM_DIR external/glm)
set(VULKAN_MEMORY_ALLOCATOR_DIR external/VulkanMemoryAllocator)
set(GLFW_DIR external/glfw)
set(TINYGLFT_DIR external/tinygltf)
set(LIBMULTISENSE_DIR external/LibMultiSense)
set(FMT_DIR external/fmt)
set(LIBTIFF_DIR external/libtiff)
set(IMGUI_DIR external/imgui)
set(KTX_DIR external/KTX-Software)
set(NLOHMANN_JSON external/json)
set(CPP_HTTPLIB external/cpp-httplib)
set(ENTT_DIR external/entt)
set(TINY_OBJ_LOADER_DIR external/tinyobjloader)
set(TINY_PLY_DIR external/tinyply)

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${GLM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${GLM_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding GLM from directory: ${GLM_DIR}")

    include_directories(SYSTEM ${GLM_DIR})

endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${VULKAN_MEMORY_ALLOCATOR_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${VULKAN_MEMORY_ALLOCATOR_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding VULKAN_MEMORY_ALLOCATOR_DIR from directory: ${VULKAN_MEMORY_ALLOCATOR_DIR}")

    add_subdirectory(${VULKAN_MEMORY_ALLOCATOR_DIR})

endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${GLFW_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${GLFW_DIR}/ not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding GLFW from directory: ${GLFW_DIR}")

    add_subdirectory(${GLFW_DIR} glfw_binary EXCLUDE_FROM_ALL)
    include_directories(SYSTEM ${GLFW_DIR}/include)
    include_directories(SYSTEM ${GLFW_DIR}/deps)
endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${TINYGLFT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINYGLFT_DIR}/ not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding TINYGLTF from directory: ${TINYGLFT_DIR}")

    set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
    set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)
    include_directories(SYSTEM ${TINYGLFT_DIR})
endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${TINY_OBJ_LOADER_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINY_OBJ_LOADER_DIR}/ not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding TINY_OBJ_LOADER_DIR from directory: ${TINY_OBJ_LOADER_DIR}")

    add_subdirectory(${TINY_OBJ_LOADER_DIR})
    include_directories(${TINY_OBJ_LOADER_DIR})

endif ()
if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${TINY_PLY_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINY_PLY_DIR}/ not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding TINY_PLY_DIR from directory: ${TINY_PLY_DIR}")
    add_subdirectory(${TINY_PLY_DIR})
    include_directories("${TINY_PLY_DIR}/source")
endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${LIBTIFF_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINYTIFF_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding libtiff from directory: ${LIBTIFF_DIR}")
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(tiff-tools OFF)
    set(tiff-tests OFF)
    set(tiff-contrib OFF)
    set(tiff-docs OFF)
    add_subdirectory(${LIBTIFF_DIR})
    include_directories(${LIBTIFF_DIR}/libtiff)
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${IMGUI_DIR}/imgui.h")
    message(FATAL_ERROR "The submodules ${IMGUI_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding IMGUI from directory: ${IMGUI_DIR}")

    set(IMGUI_DIR external/imgui)
    include_directories(SYSTEM ${IMGUI_DIR} ${IMGUI_DIR}/backends ..)
    include_directories(${PROJECT_SOURCE_DIR}/src/Viewer/ImGui/Custom) # Custom IMGUI application
    set(IMGUI_SRC
            ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/imgui.cpp
            ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_demo.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp
            src/Viewer/ImGui/Custom/imgui_user.h)
    add_library(imgui STATIC ${IMGUI_SRC})
endif ()


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${FMT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${FMT_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding FMT from directory: ${FMT_DIR}")
    include_directories(SYSTEM ${FMT_DIR}/include)
    add_subdirectory(${FMT_DIR})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${LIBMULTISENSE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${LIBMULTISENSE_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding LIBMULTISENSE_DIR from directory: ${LIBMULTISENSE_DIR}")
    include_directories(SYSTEM ${LIBMULTISENSE_DIR}/source/LibMultiSense/include)
    add_subdirectory(${LIBMULTISENSE_DIR}/source/LibMultiSense)
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${KTX_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${KTX_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding KTX from directory: ${KTX_DIR}")
    set(KTX_FEATURE_STATIC_LIBRARY ON)
    set(KTX_FEATURE_TESTS OFF)
    set(KTX_FEATURE_TOOLS OFF)
    add_subdirectory(${KTX_DIR})
    include_directories(SYSTEM ${KTX_DIR}/include)
    link_directories(${KTX_DIR}/lib)

endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${NLOHMANN_JSON}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${NLOHMANN_JSON} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding NLOHMANN_JSON from directory: ${NLOHMANN_JSON}")
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install OFF CACHE INTERNAL "")
    add_subdirectory(${NLOHMANN_JSON})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${CPP_HTTPLIB}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${CPP_HTTPLIB} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding CPP_HTTPLIB from directory: ${CPP_HTTPLIB}")
    set(HTTPLIB_REQUIRE_OPENSSL OFF)
    set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF)
    set(OPENSSL_USE_STATIC_LIBS ON)
    add_subdirectory(${CPP_HTTPLIB})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${ENTT_DIR}")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${ENTT_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding ENTT_DIR from directory: ${ENTT_DIR}")
    include_directories(${ENTT_DIR}/include/)
endif ()

# ExportScriptIncludes Generates ScriptHeader.h and Scripts.txt for automatic import of the script functionality in the viewer.
function(ExportScriptIncludes)
    set(SCRIPT_HEADER_FILE "${CMAKE_SOURCE_DIR}/Assets/Generated/ScriptHeader.h")
    set(SCRIPTS_TXT_FILE "${CMAKE_SOURCE_DIR}/Assets/Generated/Scripts.txt")

    # Remove if exists to start clean
    if(EXISTS ${SCRIPT_HEADER_FILE})
        file(REMOVE ${SCRIPT_HEADER_FILE})
    endif()

    if(EXISTS ${SCRIPTS_TXT_FILE})
        file(REMOVE ${SCRIPTS_TXT_FILE})
    endif()

    string(TIMESTAMP Today)
    file(GLOB_RECURSE SCRIPT_HEADERS RELATIVE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src/Viewer/Scripts/*.h")
    list(FILTER SCRIPT_HEADERS EXCLUDE REGEX "/ScriptSupport/")

    # Print the entire SCRIPT_HEADERS list
    message(STATUS "[INFO]: SCRIPT_HEADERS: ${SCRIPT_HEADERS}")

    # Use a set to avoid duplicates
    set(UNIQUE_SCRIPT_HEADERS)
    foreach (Src ${SCRIPT_HEADERS})
        list(FIND UNIQUE_SCRIPT_HEADERS ${Src} _index)
        if (_index EQUAL -1)
            list(APPEND UNIQUE_SCRIPT_HEADERS ${Src})
        endif()
    endforeach()

    file(WRITE ${SCRIPT_HEADER_FILE} "// Generated from CMake ${Today} \n")
    file(WRITE ${SCRIPTS_TXT_FILE} "# Generated from CMake ${Today} \n")

    foreach (Src ${UNIQUE_SCRIPT_HEADERS})
        file(APPEND ${SCRIPT_HEADER_FILE} "\#include \"${Src}\"\n")
    endforeach()

    foreach (Src ${UNIQUE_SCRIPT_HEADERS})
        string(REGEX MATCH "[^\\/]+$" var ${Src})
        string(REGEX MATCH "^[^.]+" res ${var})
        file(APPEND ${SCRIPTS_TXT_FILE} ${res} \n)
    endforeach()
    message(STATUS "ExportScriptIncludes executed successfully.")

endfunction()

function(GenerateVersionFile)
    file(WRITE ${CMAKE_SOURCE_DIR}/Assets/Generated/VersionInfo "VERSION=${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}\nSERVER=${CRL_SERVER_IP}\nPROTOCOL=${CRL_SERVER_PROTOCOL}\nDESTINATION=${CRL_SERVER_DESTINATION}\nDESTINATION_VERSIONINFO=${CRL_SERVER_VERSIONINFO_DESTINATION}\nLOG_LEVEL=${VIEWER_LOG_LEVEL}")
endfunction()

if (UNIX)
    set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/opt/multisense)
elseif (WIN32)
    set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/)
endif ()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "Set install directory to ${INSTALL_DIRECTORY}")
    set(${CMAKE_INSTALL_PREFIX} ${INSTALL_DIRECTORY}
            CACHE PATH "default install path" FORCE)
endif ()
