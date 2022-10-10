cmake_minimum_required(VERSION 3.6)
project(MultiSense-Viewer C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED true)

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

# Dear ImGui
set(IMGUI_DIR external/imgui)
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)
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

# Ini file parser SimpleIni
set(SimpleIni_DIR external/simpleini)
set(SimpleIni_SRC ${SimpleIni_DIR}/SimpleIni.h ${SimpleIni_DIR}/ConvertUTF.c ${SimpleIni_DIR}/ConvertUTF.h)
add_library(SimpleIni STATIC ${SimpleIni_SRC})

# Internal Library
set(AUTOCONNECT_SRC "internal/AutoConnect/")
include_directories(${AUTOCONNECT_SRC})
add_subdirectory(${AUTOCONNECT_SRC})

## FFMPEG
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/include)

# FMT
# Used for logging purposes
set(FMT_DIR external/fmt)  
add_subdirectory(${FMT_DIR})

# add source files
set(IMGUI_SRC
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_demo.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp
        src/imgui/custom/imgui_user.h
        src/imgui/custom/imgui_user.inl
        )
INCLUDE_DIRECTORIES(src/imgui/custom)

# LIBMULTISENSE
add_subdirectory(${MULTISENSE_SRC} EXCLUDE_FROM_ALL)


set(LIBRARIES "glfw;Vulkan::Vulkan;LibAutoConnect;fmt::fmt;SimpleIni")


file(GLOB_RECURSE SCRIPT_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "src/Scripts/Objects/*.h")
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/src/Scripts/Private/ScriptHeader.h "// Generated from Cmake\n")
foreach (src ${SCRIPT_HEADERS})
    file(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/Scripts/Private/ScriptHeader.h "\#include \"MultiSense/" ${src} "\"\n")
endforeach(src ${SCRIPT_HEADERS})
file(GLOB_RECURSE SCRIPT_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "src/Scripts/Objects/*.cpp")

set(ENGINE_SRC
        src/Core/Definitions.h
        src/Core/KeyInput.h
        src/Scripts/Private/Base.h
        src/Core/Buffer.cpp
        src/Core/Buffer.h
        src/Core/Camera.h
        src/CRLCamera/CameraConnection.cpp
        src/CRLCamera/CameraConnection.h
        src/Scripts/Private/ScriptBuilder.h
        src/Core/Texture.cpp
        src/Core/Texture.h
        src/Core/Validation.h
        src/Core/VulkanDevice.cpp
        src/Core/VulkanDevice.h
        src/Core/VulkanRenderer.cpp
        src/Core/VulkanRenderer.h
        src/Core/VulkanSwapchain.cpp
        src/Core/VulkanSwapchain.h
        src/CRLCamera/ThreadPool.h
        src/CRLCamera/CRLPhysicalCamera.cpp
        src/CRLCamera/CRLPhysicalCamera.h
        src/imgui/GuiManager.cpp
        src/imgui/GuiManager.h
        src/imgui/InteractionMenu.h
        src/imgui/Layer.h
        src/imgui/SideBar.h
        src/imgui/LayerExample.h
        src/ModelLoaders/CRLCameraModels.cpp
        src/ModelLoaders/CRLCameraModels.h
        src/ModelLoaders/glTFModel.cpp
        src/ModelLoaders/glTFModel.h
        src/Renderer/Renderer.cpp
        src/Renderer/Renderer.h

        ${SCRIPT_HEADERS}
        ${SCRIPT_SRC}

        src/Tools/Macros.h
        src/Tools/Populate.h
        src/Tools/Utils.h
        src/Tools/Logger.h
        src/Tools/Logger.cpp
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

    add_subdirectory(internal/LibMultiSenseWindows)
    add_subdirectory(internal/winpcap)


    add_library(VulkanRenderer ${ENGINE_SRC} ${IMGUI_SRC})
    target_link_libraries(VulkanRenderer  winpcap_lib packet_lib avformat avutil avcodec swscale ${LIBRARIES} MultiSenseWindows) 
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
    # MultiSense Viewer DLL and libraries
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

    # install(TARGETS MultiSense-viewer CONFIGURATIONS Release RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})


else () ## Linux

    add_library(VulkanRenderer ${ENGINE_SRC} ${IMGUI_SRC})
    target_link_libraries(VulkanRenderer ${LIBRARIES} MultiSense -lavformat -lavcodec -lavutil  -ltbb)

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