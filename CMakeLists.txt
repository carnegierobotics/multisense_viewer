cmake_minimum_required(VERSION 3.2)
project(MultiSense-Viewer C CXX)


#set(BUILD_SHARED_LIBS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set install directory if not specified
set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")
set(ARCHITECTURE "amd64")

# Set install directory if not specified
set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/opt/multisense)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message("   INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
    set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}"
            CACHE PATH "default install path" FORCE)
endif()

include(cmake/Common.cmake)
find_package(Vulkan REQUIRED)

option(ENABLE_ALL_WARNINGS_GCC "Compile with all warnings" OFF)
include_directories(include)
include_directories(Assets)
include_directories(external)
include_directories(external/ffmpeg/include)
include_directories(${Vulkan_INCLUDE_DIRS})
cmake_language(CALL ExportScriptIncludes)

set(ENGINE_SRC
        src/Core/Buffer.cpp
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
        src/CRLCamera/CameraConnection.cpp
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
        src/Scripts/Objects/Video/Previews/Quad/Four.cpp)

set(LIBRARIES "Vulkan::Vulkan;glfw;SimpleIni;fmt::fmt;imgui;TinyTIFF;MultiSense;LibAutoConnect")

# Copy Assets folders to output directory
file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/Assets/*")
file(COPY ${ASSETS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Assets)

# Generate empty configuration files.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/imgui.ini "")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/crl.ini "")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/logger.log "")

if (UNIX) ## Linux
    set(AUTOCONNECT_EXECUTABLE AutoConnect)

    add_library(VulkanRenderer STATIC ${ENGINE_SRC})
    target_link_libraries(VulkanRenderer ${LIBRARIES})

    add_library(ScriptLibrary STATIC ${SCRIPTS_LIB_SRC})
    target_link_libraries(ScriptLibrary VulkanRenderer  -lavutil -lswscale)

    add_executable(MultiSense-Viewer src/main.cpp)
    target_link_libraries(MultiSense-Viewer VulkanRenderer ScriptLibrary)
    add_dependencies(MultiSense-Viewer AutoConnect)

    install(TARGETS MultiSense-Viewer RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})
    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/imgui.ini
            ${CMAKE_CURRENT_BINARY_DIR}/crl.ini
            ${CMAKE_CURRENT_BINARY_DIR}/logger.log
            CONFIGURATIONS Release DESTINATION   ${INSTALL_DIRECTORY}
            )
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/AutoConnect CONFIGURATIONS Release DESTINATION   ${INSTALL_DIRECTORY})
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/AutoConnectLauncher.sh CONFIGURATIONS Release DESTINATION  ${INSTALL_DIRECTORY})


endif ()

if (WIN32)
    set(AUTOCONNECT_EXECUTABLE AutoConnect.exe)

    if (MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
    endif ()

    # Include directory for WinPcap
    include_directories(internal/AutoConnect/include/WinPcap)

    add_library(avutil STATIC IMPORTED GLOBAL)
    set_property(TARGET avutil APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(avutil PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
            IMPORTED_LOCATION_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/avutil.lib")

    add_library(swscale STATIC IMPORTED GLOBAL)
    set_property(TARGET swscale APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(swscale PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
            IMPORTED_LOCATION_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/swscale.lib")

    add_library(VulkanRenderer STATIC ${ENGINE_SRC})
    target_link_libraries(VulkanRenderer ${LIBRARIES})

    add_library(ScriptLibrary STATIC ${SCRIPTS_LIB_SRC})
    target_link_libraries(ScriptLibrary VulkanRenderer  avutil swscale)

    add_executable(MultiSense-Viewer WIN32 src/main.cpp)
    target_link_libraries(MultiSense-Viewer VulkanRenderer ScriptLibrary)
    add_dependencies(MultiSense-Viewer AutoConnect)

    # create a list of files to copy
    set(THIRD_PARTY_DLLS
            #FFMPEG
            #${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avcodec-59.dll
            #${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avdevice-59.dll
            #${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avfilter-8.dll
            #${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avformat-59.dll
            #${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/swresample-4.dll
            #${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/postproc-56.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/swscale-6.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avutil-57.dll
            )

    # do the copying
    foreach (file_i ${THIRD_PARTY_DLLS})
        add_custom_command(
                TARGET VulkanRenderer
                POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                ARGS -E copy ${file_i} ${CMAKE_BINARY_DIR}/
        )
    endforeach (file_i)
    add_custom_command(TARGET MultiSense-Viewer
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/external/LibMultiSense/source/LibMultiSense/MultiSense.dll ${CMAKE_BINARY_DIR}

            # Windows AutoConnect WinPcap dlls
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/internal/AutoConnect/packet.dll ${CMAKE_BINARY_DIR}/
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/internal/AutoConnect/wpcap.dll ${CMAKE_BINARY_DIR}/

            )
    ### INSTALL TARGETS ###
    # VkRender Viewer DLL and libraries
    #install(TARGETS MultiSense-Viewer CONFIGURATIONS Release RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    #install(TARGETS MultiSense-Viewer CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    #install(TARGETS MultiSense-Viewer CONFIGURATIONS Release LIBRARY DESTINATION ${INSTALL_DIRECTORY})

    # LibMultiSense and WinPCap dll
    install(FILES
            ${CMAKE_BINARY_DIR}/Release/MultiSense.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/internal/winpcap/Packet.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/internal/winpcap/wpcap.dll
            CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})

    ## dll's from fmmpeg
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/ DESTINATION ${INSTALL_DIRECTORY}
            FILES_MATCHING PATTERN "*.dll")

    # Copy Assets
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})

    if (MSVC)
        if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
            message("\n\t[ WARNING ]\n\n\tCMake version lower than 3.6.\n\n\t - Please update CMake and rerun; OR\n\t - Manually set 'GLFW-CMake-starter' as StartUp Project in Visual Studio.\n")
        else ()
            set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MultiSense-Viewer)
            #set_property(TARGET MultiSense-Viewer PROPERTY CXX_STANDARD 20)
        endif ()
    endif ()

    file(WRITE ${INSTALL_DIRECTORY}/winreg.ini "")
endif ()


add_custom_command(TARGET MultiSense-Viewer
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/internal/AutoConnect/${AUTOCONNECT_EXECUTABLE} ${CMAKE_BINARY_DIR}/
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/Assets/Tools/Ubuntu/AutoConnectLauncher.sh ${CMAKE_BINARY_DIR}/
        )