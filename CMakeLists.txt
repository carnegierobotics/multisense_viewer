cmake_minimum_required(VERSION 3.6)
project(MultiSense-Viewer C CXX)


set(BUILD_SHARED_LIBS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/Common.cmake)

find_package(Vulkan REQUIRED)

option(ENABLE_ALL_WARNINGS "Build unit tests." ON)

if (UNIX AND ${ENABLE_ALL_WARNINGS})
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -Wshadow -Wpointer-arith -Wuninitialized")
    # Not enabled flags but could be nice to enable
    # -Wundef -Wcast-qual  -Wdouble-promotion
endif()


set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")
set(ARCHITECTURE "amd64")



# GLFW
set(GLFW_DIR external/glfw) # Set this to point to an up-to-date GLFW repo  option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
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


set(TinyTIFF_BUILD_STATIC_LIBS ON)
set(TinyTIFF_BUILD_SHARED_LIBS OFF)
include_directories(${CMAKE_BINARY_DIR}/external/TinyTIFF/src)
set(TinyTIFF_BUILD_DECORATE_LIBNAMES_WITH_BUILDTYPE OFF)
set(TinyTIFF_SRC external/TinyTIFF)
add_subdirectory(${TinyTIFF_SRC})
  
#include headers for LibMultiSense. Implementation differs for Win/Linux so further linking is done separately
if (WIN32)
set(BUILD_SHARED_LIBS TRUE)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/) # Set install prefix to LibMultiSense installs to our build directory
endif()
set(MULTISENSE_SRC "external/LibMultiSense/source/LibMultiSense")
include_directories(${MULTISENSE_SRC})
add_subdirectory(${MULTISENSE_SRC})
if (WIN32)
    set(BUILD_SHARED_LIBS False)
endif()
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
        ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_demo.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp ${IMGUI_FILEDIALOG_DIR}/ImGuiFileDialog.cpp
        include/Viewer/ImGui/Custom/imgui_user.h
        )

include_directories(include)
include_directories(Assets)
include_directories(external)
include_directories(internal/AutoConnect/include)

cmake_language(CALL ExportScriptIncludes)

set(ENGINE_SRC
        src/Core/Buffer.cpp
        src/CRLCamera/CameraConnection.cpp
        src/Core/Texture.cpp
        src/Core/VulkanDevice.cpp
        src/Core/VulkanRenderer.cpp
        src/Core/VulkanSwapchain.cpp
        src/CRLCamera/CRLPhysicalCamera.cpp
        src/ImGui/GuiManager.cpp
        src/ModelLoaders/CRLCameraModels.cpp
        src/ModelLoaders/GLTFModel.cpp
        src/Renderer/Renderer.cpp
        src/Tools/Logger.cpp
        src/Tools/StbInclude.cpp
        )

set(SCRIPTS_LIB_SRC
        src/Scripts/Objects/MultiSenseCamera.cpp
        src/Scripts/Objects/Example.cpp
        src/Scripts/Objects/Pointcloud/PointCloud.cpp
        src/Scripts/Objects/SceneGizmos/Gizmos.cpp
        src/Scripts/Objects/Video/RecordExposure.cpp
        src/Scripts/Objects/Video/RecordFrames.cpp
        src/Scripts/Objects/Video/Previews/Single/SingleLayout.cpp
        src/Scripts/Objects/Video/Previews/Double/DoubleTop.cpp
        src/Scripts/Objects/Video/Previews/Double/DoubleBot.cpp
        src/Scripts/Objects/Video/Previews/Quad/One.cpp
        src/Scripts/Objects/Video/Previews/Quad/Two.cpp
        src/Scripts/Objects/Video/Previews/Quad/Three.cpp
        src/Scripts/Objects/Video/Previews/Quad/Four.cpp
        )

set(LIBRARIES "glfw;Vulkan::Vulkan;fmt::fmt;LibAutoConnect;SimpleIni;MultiSense;TinyTIFF")

if (WIN32)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")

    if (MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
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

    add_library(MultiSenseLib STATIC IMPORTED GLOBAL)
    set_property(TARGET MultiSenseLib APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(MultiSenseLib PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
    IMPORTED_LOCATION_DEBUG  "${CMAKE_BINARY_DIR}/external/LibMultiSense/source/LibMultiSense/MultiSense.lib")


    add_subdirectory(internal/winpcap)

    add_library(VulkanRenderer STATIC ${ENGINE_SRC} ${IMGUI_SRC})
    target_link_libraries(VulkanRenderer  winpcap_lib packet_lib avformat avutil avcodec swscale ${LIBRARIES} LibAutoConnect)
    # LibMultiSense on Windows. Use precompiled library from a manual step

    add_library(ScriptLibrary STATIC ${SCRIPTS_LIB_SRC})
    target_link_libraries(ScriptLibrary VulkanRenderer)

    add_executable(MultiSense-viewer WIN32 main.cpp)
    target_link_libraries(MultiSense-viewer VulkanRenderer ScriptLibrary)

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
    install(TARGETS MultiSense-viewer CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS MultiSense-viewer CONFIGURATIONS Release LIBRARY DESTINATION ${INSTALL_DIRECTORY})

    # LibMultiSense and WinPCap dll
    install(FILES  
     ${CMAKE_BINARY_DIR}/Release/MultiSense.dll
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

    file(WRITE ${INSTALL_DIRECTORY}/winreg.ini "")

else () ## Linux
    add_library(VulkanRenderer STATIC ${ENGINE_SRC} ${IMGUI_SRC})
    target_link_libraries(VulkanRenderer ${LIBRARIES} MultiSense -lavformat -lavcodec -lavutil  -lswscale -ltbb)

    add_library(ScriptLibrary STATIC ${SCRIPTS_LIB_SRC})
    target_link_libraries(ScriptLibrary VulkanRenderer )

    add_executable(MultiSense-viewer src/main.cpp)
    target_link_libraries(MultiSense-viewer VulkanRenderer ScriptLibrary)


    set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/opt/multisense)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/opt/multisense)

    install(TARGETS MultiSense-viewer CONFIGURATIONS Release RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS MultiSense-viewer CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS MultiSense-viewer CONFIGURATIONS Release LIBRARY DESTINATION ${INSTALL_DIRECTORY})
    MESSAGE("[INFO] COPYING ASSETS TO RUNTIME DESTINATION: ${INSTALL_DIRECTORY}")
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})
endif ()

# Generate empty configuration files
file(WRITE ${INSTALL_DIRECTORY}/imgui.ini "")
file(WRITE ${INSTALL_DIRECTORY}/crl.ini "")
file(WRITE ${INSTALL_DIRECTORY}/logger.log "")

# Copy dependency folders to output directory
add_custom_target(copy_assets COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/Assets ${CMAKE_CURRENT_BINARY_DIR}/Assets)
add_dependencies(MultiSense-viewer copy_assets)
file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/Assets/*")
file(COPY ${ASSETS} DESTINATION  ${CMAKE_CURRENT_BINARY_DIR}/Assets)