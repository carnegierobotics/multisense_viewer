cmake_minimum_required(VERSION 3.6)
project(MultiSense-Viewer C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED true)

if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -Wshadow -Wpointer-arith -Wuninitialized ")
    # Not enabled flags but could be nice to enable
    # -Wundef -Wcast-qual  -Wdouble-promotion
endif (UNIX)



set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")
set(ARCHITECTURE "amd64")

# Libraries
find_package(Vulkan REQUIRED)

# GLFW
set(GLFW_DIR external/glfw) # Set this to point to an up-to-date GLFW repo
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
add_subdirectory(${GLFW_DIR} glfw_binary EXCLUDE_FROM_ALL)
include_directories(${GLFW_DIR}/include)

# GLM
set(GLM_DIR external/glm)
include_directories(${GLM_DIR})
# Use vulkan headers from glfw:
include_directories(${GLFW_DIR}/deps)

# tiny-glTF header only loader
set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)
include_directories(external/tinygltf/)

#include headers for LibMultiSense. Implementation differs for Win/Linux so further linking is done separately
set(MULTISENSE_SRC external/LibMultiSense/source/LibMultiSense)
include_directories(${MULTISENSE_SRC})
add_subdirectory(${MULTISENSE_SRC})

# Ini file parser SimpleIni
set(SimpleIni_DIR external/simpleini)
set(SimpleIni_SRC ${SimpleIni_DIR}/SimpleIni.h ${SimpleIni_DIR}/ConvertUTF.c ${SimpleIni_DIR}/ConvertUTF.h)
add_library(SimpleIni STATIC ${SimpleIni_SRC})

# FMT
# Used for logging purposes
set(FMT_DIR external/fmt)  
add_subdirectory(${FMT_DIR})

# Internal Library
set(AUTOCONNECT_SRC "internal/AutoConnect/")
include_directories(${AUTOCONNECT_SRC})
add_subdirectory(${AUTOCONNECT_SRC})

## FFMPEG
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/include)


# add ImGUI source files
#Also Include ImGUi Custom FileDialog Extension
# Dear ImGui
set(IMGUI_DIR external/imgui)
set(IMGUI_FILEDIALOG_DIR external/ImGuiFileDialog)

include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)
include_directories(${IMGUI_FILEDIALOG_DIR})

set(IMGUI_SRC
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_demo.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp
        Src/imgui/custom/imgui_user.h
        Src/imgui/custom/imgui_user.inl ${IMGUI_FILEDIALOG_DIR}/ImGuiFileDialog.cpp
        )
INCLUDE_DIRECTORIES(Src/imgui/custom)
set(LIBRARIES "glfw;Vulkan::Vulkan;fmt::fmt;LibAutoConnect;SimpleIni;MultiSense")

file(GLOB_RECURSE SCRIPT_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "Src/Scripts/Objects/*.h")
string(TIMESTAMP Today)
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/Assets/Generated/ScriptHeader.h "// Generated from Cmake ${Today} \n")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/Assets/Generated/Scripts.txt "# Generated from Cmake ${Today} \n")

foreach (Src ${SCRIPT_HEADERS})
    file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/Assets/Generated/ScriptHeader.h "\#include \"MultiSense/" ${Src} "\"\n")
endforeach(Src ${SCRIPT_HEADERS})

foreach (Src ${SCRIPT_HEADERS})
            string(REGEX MATCH "[^\\/]+$" var ${Src})
            string(REGEX MATCH "^[^.]+" res ${var})
            file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/Assets/Generated/Scripts.txt ${res} \n)
endforeach(Src ${SCRIPT_HEADERS})

file(GLOB_RECURSE SCRIPT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "Src/Scripts/Objects/*.cpp")

set(ENGINE_SRC
        Src/Core/Definitions.h
        Src/Core/KeyInput.h
        Src/Scripts/Private/Base.h
        Src/Core/Buffer.cpp
        Src/Core/Buffer.h
        Src/Core/Camera.h
        Src/CRLCamera/CameraConnection.cpp
        Src/CRLCamera/CameraConnection.h
        Src/Scripts/Private/ScriptBuilder.h
        Src/Core/Texture.cpp
        Src/Core/Texture.h
        Src/Core/Validation.h
        Src/Core/VulkanDevice.cpp
        Src/Core/VulkanDevice.h
        Src/Core/VulkanRenderer.cpp
        Src/Core/VulkanRenderer.h
        Src/Core/VulkanSwapchain.cpp
        Src/Core/VulkanSwapchain.h
        Src/CRLCamera/ThreadPool.h
        Src/CRLCamera/CRLPhysicalCamera.cpp
        Src/CRLCamera/CRLPhysicalCamera.h
        Src/imgui/GuiManager.cpp
        Src/imgui/GuiManager.h
        Src/imgui/InteractionMenu.h
        Src/imgui/Layer.h
        Src/imgui/SideBar.h
        Src/imgui/LayerExample.h
        Src/ModelLoaders/CRLCameraModels.cpp
        Src/ModelLoaders/CRLCameraModels.h
        Src/ModelLoaders/glTFModel.cpp
        Src/ModelLoaders/glTFModel.h
        Src/Renderer/Renderer.cpp
        Src/Renderer/Renderer.h
        Src/Tools/Macros.h
        Src/Tools/Populate.h
        Src/Tools/Utils.h
        Src/Tools/Logger.h
        Src/Tools/Logger.cpp
        Src/Tools/stb_include.cpp

        ${SCRIPT_HEADERS}
        ${SCRIPT_SRC}
        )

if (WIN32)

    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
    endif ()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")

    if (MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(BUILD_SHARED_LIBS TRUE)
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup")
    endif ()


    add_library(avcodec STATIC IMPORTED GLOBAL)
    set_property(TARGET avcodec APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(avcodec PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
    IMPORTED_LOCATION_RELEASE  "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/avcodec.lib")
     

    add_library(avutil STATIC IMPORTED GLOBAL)
    set_property(TARGET avutil APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(avutil PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
    IMPORTED_LOCATION_DEBUG  "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/avutil.lib")

    add_library(avformat STATIC IMPORTED GLOBAL)
    set_property(TARGET avformat APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(avformat PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
    IMPORTED_LOCATION_DEBUG  "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/avformat.lib")

    add_library(swscale STATIC IMPORTED GLOBAL)
    set_property(TARGET swscale APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(swscale PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
    IMPORTED_LOCATION_DEBUG  "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/swscale.lib")

    add_subdirectory(internal/winpcap)
    add_subdirectory(internal/LibMultiSenseWindows)

    add_library(VulkanRenderer STATIC ${ENGINE_SRC} ${IMGUI_SRC})
    target_link_libraries(VulkanRenderer  winpcap_lib packet_lib avformat avutil avcodec swscale ${LIBRARIES} LibAutoConnect MultiSenseWindows)
    # LibMultiSense on Windows. Use precompiled library from a manual step


    add_executable(MultiSense-viewer WIN32 main.cpp)
    target_link_libraries(MultiSense-viewer VulkanRenderer)


    # create a list of files to copy
    set( THIRD_PARTY_DLLS
    #FFMPEG
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avcodec-59.dll
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avdevice-59.dll
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avfilter-8.dll
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avformat-59.dll
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avutil-57.dll
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/swresample-4.dll
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/swscale-6.dll
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/postproc-56.dll
    #LibMultiSense
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/LibMultiSenseWindows/${CMAKE_BUILD_TYPE}/MultiSense.dll
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/LibMultiSenseWindows/${CMAKE_BUILD_TYPE}/MultiSense.lib
    #WinPCap
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/winpcap/Packet.dll
    ${CMAKE_CURRENT_SOURCE_DIR}/internal/winpcap/wpcap.dll
    )

    # do the copying
    foreach( file_i ${THIRD_PARTY_DLLS})
        add_custom_command(
        TARGET VulkanRenderer
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${file_i} ${CMAKE_BINARY_DIR}/
    )
endforeach( file_i )

    ### INSTALL TARGETS ###
    set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE})
    # VkRender Viewer DLL and libraries
    install(TARGETS MultiSense-viewer CONFIGURATIONS Release RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS VulkanRenderer CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS VulkanRenderer CONFIGURATIONS Release LIBRARY DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS VulkanRenderer CONFIGURATIONS Release RUNTIME DESTINATION ${INSTALL_DIRECTORY})

    # LibMultiSense and WinPCap dll
    install(FILES  
     ${CMAKE_CURRENT_SOURCE_DIR}/internal/LibMultiSenseWindows/${CMAKE_BUILD_TYPE}/MultiSense.dll
     ${CMAKE_CURRENT_SOURCE_DIR}/internal/LibMultiSenseWindows/${CMAKE_BUILD_TYPE}/MultiSense.lib
     ${CMAKE_CURRENT_SOURCE_DIR}/internal/winpcap/Packet.dll
     ${CMAKE_CURRENT_SOURCE_DIR}/internal/winpcap/wpcap.dll
     CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})

     ## dll's from fmmpeg
     install(DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/ DESTINATION ${INSTALL_DIRECTORY}
    FILES_MATCHING PATTERN "*.dll")

    # Copy Assets
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})

    if (MSVC)
        if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
            message("\n\t[ WARNING ]\n\n\tCMake version lower than 3.6.\n\n\t - Please update CMake and rerun; OR\n\t - Manually set 'GLFW-CMake-starter' as StartUp Project in Visual Studio.\n")
        else ()
            set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MultiSense-viewer)
            set_property(TARGET MultiSense-viewer PROPERTY CXX_STANDARD 20)
        endif ()
    endif ()

    # install(TARGETS VkRender-viewer CONFIGURATIONS Release RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})


else () ## Linux


    add_library(VulkanRenderer ${ENGINE_SRC} ${IMGUI_SRC})
    target_link_libraries(VulkanRenderer ${LIBRARIES} MultiSense -lavformat -lavcodec -lavutil  -lswscale -ltbb)

    add_executable(MultiSense-viewer main.cpp)
    target_link_libraries(MultiSense-viewer VulkanRenderer)


    set(RUNTIME_DESTINATION ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/opt/multisense)
    set(LIBRARY_DESTINATION ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/opt/multisense)
    set(ASSETS_DESTINATION ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/opt/multisense)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/opt/multisense)

    install(TARGETS MultiSense-viewer CONFIGURATIONS Release RUNTIME DESTINATION ${RUNTIME_DESTINATION})
    install(TARGETS VulkanRenderer CONFIGURATIONS Release ARCHIVE DESTINATION ${LIBRARY_DESTINATION})
    install(TARGETS VulkanRenderer CONFIGURATIONS Release LIBRARY DESTINATION ${LIBRARY_DESTINATION})


    MESSAGE("[INFO] COPYING ASSETS TO RUNTIME DESTINATION: ${ASSETS_DESTINATION}")
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Assets CONFIGURATIONS Release DESTINATION ${ASSETS_DESTINATION})


endif ()
# Copy dependency folders to output directory
add_custom_target(copy_assets COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/Assets ${CMAKE_CURRENT_BINARY_DIR}/Assets)
add_dependencies(MultiSense-viewer copy_assets)
file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/Assets/*")
file(COPY ${ASSETS} DESTINATION  ${CMAKE_CURRENT_BINARY_DIR}/Assets)