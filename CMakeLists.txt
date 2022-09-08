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

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXE_LINKER_FLAGS "-static")

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

# Internal Library # Parameter for Autoconnect Library
set(LIBMULTISENSE_SKIP True)
set(TOP_LEVEL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(AUTOCONNECT_SRC internal/AutoConnect/)
include_directories(${AUTOCONNECT_SRC})
add_subdirectory(${AUTOCONNECT_SRC})

## FFMPEG
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/include)

# FMT
# Used for logging purposes
set(FMT_DIR external/fmt) 
add_subdirectory(${FMT_DIR})

set(LIBRARIES "glfw;Vulkan::Vulkan;LibAutoConnect;fmt::fmt")

# add source files
set(IMGUI_SRC
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_demo.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp
        src/imgui/custom/imgui_user.h
        src/imgui/custom/imgui_user.inl
        )
INCLUDE_DIRECTORIES(src/imgui/custom)

set(ENGINE_SRC
        src/core/Definitions.h
        src/core/Base.h
        src/core/Buffer.cpp
        src/core/Buffer.h
        src/core/Camera.h
        src/core/CameraConnection.cpp
        src/core/CameraConnection.h
        src/core/ScriptBuilder.h
        src/core/Texture.cpp
        src/core/Texture.h
        src/core/Validation.h
        src/core/VulkanDevice.cpp
        src/core/VulkanDevice.h
        src/core/VulkanRenderer.cpp
        src/core/VulkanRenderer.h
        src/core/VulkanSwapchain.cpp
        src/core/VulkanSwapchain.h
        src/crl_camera/CRLBaseInterface.h
        src/crl_camera/CRLPhysicalCamera.cpp
        src/crl_camera/CRLPhysicalCamera.h
        src/crl_camera/CRLVirtualCamera.cpp
        src/crl_camera/CRLVirtualCamera.h
        src/imgui/GuiManager.cpp
        src/imgui/GuiManager.h
        src/imgui/InteractionMenu.h
        src/imgui/Layer.h
        src/imgui/SideBar.h
        src/model_loaders/CRLCameraModels.cpp
        src/model_loaders/CRLCameraModels.h
        src/model_loaders/glTFModel.cpp
        src/model_loaders/glTFModel.h
        src/model_loaders/MeshModel.cpp
        src/model_loaders/MeshModel.h
        src/Renderer/Renderer.cpp
        src/Renderer/Renderer.h
        src/scripts/Example.cpp
        src/scripts/Example.h
        src/scripts/objects/LightSource.cpp
        src/scripts/objects/LightSource.h
        src/scripts/pointcloud/PointCloud.cpp
        src/scripts/pointcloud/PointCloud.h
        src/scripts/pointcloud/VirtualPointCloud.cpp
        src/scripts/pointcloud/VirtualPointCloud.h
        src/scripts/video/physical/DefaultPreview.cpp
        src/scripts/video/physical/DefaultPreview.h
        src/scripts/video/physical/RightPreview.cpp
        src/scripts/video/physical/RightPreview.h
        src/scripts/video/physical/DisparityPreview.cpp
        src/scripts/video/physical/DisparityPreview.h
        src/scripts/video/physical/AuxiliaryPreview.cpp
        src/scripts/video/physical/AuxiliaryPreview.h
        src/scripts/video/virtual/LeftImager.cpp
        src/scripts/video/virtual/LeftImager.h
        src/scripts/video/virtual/RightImager.cpp
        src/scripts/video/virtual/RightImager.h
        src/scripts/video/virtual/AuxImager.cpp
        src/scripts/video/virtual/AuxImager.h
        src/tools/Macros.h
        src/tools/Populate.h
        src/tools/Utils.h
        src/tools/Logger.h
        src/tools/Logger.cpp
        src/scripts/video/physical/ScriptHeader.h
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
    target_link_libraries(VulkanRenderer MultiSenseWindows winpcap_lib packet_lib ${LIBRARIES} avformat avutil avcodec swscale) #avformat avcodec avutil swscale random_seed)
    # LibMultiSense on Windows. Use precompiled library from a manual step
    add_custom_command(TARGET VulkanRenderer
            PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/internal/LibMultiSenseWindows/MultiSense.dll  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/internal/LibMultiSenseWindows/MultiSense.lib  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/internal/winpcap/Packet.dll  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/internal/winpcap/wpcap.dll  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/

            )



    add_executable(MultiSense-viewer WIN32 main.cpp)
    target_link_libraries(MultiSense-viewer VulkanRenderer)

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

    # LIBMULTISENSE
    add_subdirectory(${MULTISENSE_SRC} EXCLUDE_FROM_ALL)

    add_library(VulkanRenderer ${ENGINE_SRC} ${IMGUI_SRC})
    target_link_libraries(VulkanRenderer ${LIBRARIES} MultiSense)

    add_executable(MultiSense-viewer main.cpp)
    target_link_libraries(MultiSense-viewer VulkanRenderer -lavformat -lavcodec -lavutil -ltbb)


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


## TODO: COPY dependencies with executables (shaders etc..)
## TODO COPY FFMPEG DLLs for windows build
# Copy dependency folders to output directory
add_custom_target(copy_assets COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/Assets ${CMAKE_CURRENT_BINARY_DIR}/Assets)
add_dependencies(MultiSense-viewer copy_assets)
file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/Assets/*")
file(COPY ${ASSETS} DESTINATION  ${CMAKE_CURRENT_BINARY_DIR}/Assets)

# Copy DLLs
if (WIN32)
#add_custom_target(copy_assets COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin ${CMAKE_CURRENT_BINARY_DIR}/Assets)
#add_dependencies(MultiSense-viewer copy_assets)
file(GLOB FFMPEG_DLL "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/*.dll")
file(COPY ${FFMPEG_DLL} DESTINATION  ${CMAKE_CURRENT_BINARY_DIR})
endif()
