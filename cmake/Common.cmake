# CMAKE FILE DESCRIPTION
# - Sets build type to Release if none was specified
# - Pulls submodules into /external folder
# - Adds each submodule to project add_subdirectory/include_directory
# - Adds internal library Autoconnect
# - Option to enabled all warnings when compiling with GCC


find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
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

# Include Submodules into project.
# Check if exists or display fatal error
set(GLM_DIR external/glm)
set(GLFW_DIR external/glfw)
set(TINYGLFT_DIR external/tinygltf)
set(FMT_DIR external/fmt)
set(LIBTIFF_DIR external/libtiff)
set(LIBMULTISENSE_DIR external/LibMultiSense)
set(SIMPLEINI_DIR external/simpleini)
set(IMGUI_DIR external/imgui)
set(AUTOCONNECT_DIR internal/AutoConnect)


if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${GLM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${GLM_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding GLM from directory: ${GLM_DIR}")

    add_subdirectory(${GLM_DIR})
    include_directories(SYSTEM ${GLM_DIR})
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

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${FMT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${FMT_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding FMT from directory: ${FMT_DIR}")

    add_subdirectory(${FMT_DIR})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${LIBTIFF_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${TINYTIFF_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding libtiff from directory: ${LIBTIFF_DIR}")
    add_subdirectory(${LIBTIFF_DIR})
    include_directories(${LIBTIFF_DIR}/libtiff)
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${LIBMULTISENSE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${LIBMULTISENSE_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding LibMultiSense from directory: ${LIBMULTISENSE_DIR}")
    set(MULTISENSE_BUILD_UTILITIES OFF)
    if (WIN32)
        set(BUILD_SHARED_LIBS ON)
    endif()
    include_directories(SYSTEM ${LIBMULTISENSE_DIR}/source/LibMultiSense)
    add_subdirectory(${LIBMULTISENSE_DIR}/source/LibMultiSense)
    if (WIN32)
        set(BUILD_SHARED_LIBS OFF)
    endif()

endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${SIMPLEINI_DIR}/SimpleIni.h")
    message(FATAL_ERROR "The submodules ${SIMPLEINI_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding SIMPLEINI from directory: ${SIMPLEINI_DIR}")

    set(SimpleIni_SRC ${SIMPLEINI_DIR}/SimpleIni.h ${SIMPLEINI_DIR}/ConvertUTF.c ${SIMPLEINI_DIR}/ConvertUTF.h)
    add_library(SimpleIni STATIC ${SimpleIni_SRC})
    set_target_properties(SimpleIni PROPERTIES LINKER_LANGUAGE CXX)
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${IMGUI_DIR}/imgui.h")
    message(FATAL_ERROR "The submodules ${IMGUI_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding IMGUI from directory: ${IMGUI_DIR}")

    set(IMGUI_DIR external/imgui)
    set(IMGUI_FILEDIALOG_DIR external/ImGuiFileDialog)
    include_directories(SYSTEM ${IMGUI_DIR} ${IMGUI_DIR}/backends ..)
    set(IMGUI_SRC
            ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/imgui.cpp
            ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_demo.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp ${IMGUI_FILEDIALOG_DIR}/ImGuiFileDialog.cpp
            include/Viewer/ImGui/Custom/imgui_user.h)
    add_library(imgui STATIC ${IMGUI_SRC})
endif ()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${AUTOCONNECT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "The submodules ${PROJECT_SOURCE_DIR}/${AUTOCONNECT_DIR} not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
else ()
    message("[INFO] Adding AUTOCONNECT from directory: ${AUTOCONNECT_DIR}")
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(BUILD_STANDALONE OFF)
    add_subdirectory(${AUTOCONNECT_DIR})
endif ()

# ExportScriptIncludes Generates ScriptHeader.h and Scripts.txt for automatic import of the script functionality in the viewer.
function(ExportScriptIncludes)
    string(TIMESTAMP Today)
    file(GLOB_RECURSE SCRIPT_HEADERS RELATIVE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/Viewer/Scripts/Objects/*.h")
    file(WRITE ${CMAKE_SOURCE_DIR}/Assets/Generated/ScriptHeader.h "// Generated from Cmake ${Today} \n")
    file(WRITE ${CMAKE_SOURCE_DIR}/Assets/Generated/Scripts.txt "# Generated from Cmake ${Today} \n")
    foreach (Src ${SCRIPT_HEADERS})
        file(APPEND ${CMAKE_SOURCE_DIR}/Assets/Generated/ScriptHeader.h "\#include \"${Src}\"\n")
    endforeach (Src ${SCRIPT_HEADERS})

    foreach (Src ${SCRIPT_HEADERS})
        string(REGEX MATCH "[^\\/]+$" var ${Src})
        string(REGEX MATCH "^[^.]+" res ${var})
        file(APPEND ${CMAKE_SOURCE_DIR}/Assets/Generated/Scripts.txt ${res} \n)
    endforeach (Src ${SCRIPT_HEADERS})
endfunction()


if (UNIX)
    set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/opt/multisense)
elseif(WIN32)
    set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/)
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "Set install directory to ${INSTALL_DIRECTORY}")
    set(${CMAKE_INSTALL_PREFIX} ${INSTALL_DIRECTORY}
            CACHE PATH "default install path" FORCE)
endif()
