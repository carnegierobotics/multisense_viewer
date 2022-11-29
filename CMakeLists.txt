cmake_minimum_required(VERSION 3.2)
project(MultiSense-Viewer C CXX)


set(BUILD_SHARED_LIBS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set install directory if not specified
set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")
set(ARCHITECTURE "amd64")
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
        src/Tools/StbInclude.cpp)

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

set(LIBRARIES "glfw;Vulkan::Vulkan;fmt::fmt;LibAutoConnect;SimpleIni;MultiSense;TinyTIFF;imgui")

# Copy Assets folders to output directory
file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/Assets/*")
file(COPY ${ASSETS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Assets)

# Generate empty configuration files.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/imgui.ini "")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/crl.ini "")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/logger.log "")

if (UNIX) ## Linux
    add_library(VulkanRenderer STATIC ${ENGINE_SRC})
    target_link_libraries(VulkanRenderer ${LIBRARIES} -lavutil -lswscale)

    add_library(ScriptLibrary STATIC ${SCRIPTS_LIB_SRC})
    target_link_libraries(ScriptLibrary VulkanRenderer)

    add_executable(MultiSense-Viewer src/main.cpp)
    target_link_libraries(MultiSense-Viewer VulkanRenderer ScriptLibrary)
    add_dependencies(MultiSense-Viewer AutoConnect)

    install(TARGETS MultiSense-Viewer RUNTIME DESTINATION app/)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY}/app)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/imgui.ini CONFIGURATIONS Release DESTINATION   ${INSTALL_DIRECTORY}/app)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/crl.ini CONFIGURATIONS Release DESTINATION   ${INSTALL_DIRECTORY}/app)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/logger.log CONFIGURATIONS Release DESTINATION   ${INSTALL_DIRECTORY}/app)
endif ()

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
            IMPORTED_LOCATION_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/avcodec.lib")

    add_library(avutil STATIC IMPORTED GLOBAL)
    set_property(TARGET avutil APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(avutil PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
            IMPORTED_LOCATION_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/avutil.lib")

    add_library(avformat STATIC IMPORTED GLOBAL)
    set_property(TARGET avformat APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(avformat PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
            IMPORTED_LOCATION_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/avformat.lib")

    add_library(swscale STATIC IMPORTED GLOBAL)
    set_property(TARGET swscale APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(swscale PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
            IMPORTED_LOCATION_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/swscale.lib")

    add_library(MultiSenseLib STATIC IMPORTED GLOBAL)
    set_property(TARGET MultiSenseLib APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_target_properties(MultiSenseLib PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
            IMPORTED_LOCATION_DEBUG "${CMAKE_BINARY_DIR}/external/LibMultiSense/source/LibMultiSense/MultiSense.lib")


    add_subdirectory(internal/winpcap)

    add_library(VulkanRenderer STATIC ${ENGINE_SRC} ${IMGUI_SRC})
    target_link_libraries(VulkanRenderer winpcap_lib packet_lib avformat avutil avcodec swscale ${LIBRARIES} LibAutoConnect)
    # LibMultiSense on Windows. Use precompiled library from a manual step

    add_library(ScriptLibrary STATIC ${SCRIPTS_LIB_SRC})
    target_link_libraries(ScriptLibrary VulkanRenderer)

    add_executable(MultiSense-Viewer WIN32 main.cpp)
    target_link_libraries(MultiSense-Viewer VulkanRenderer ScriptLibrary)

    # create a list of files to copy
    set(THIRD_PARTY_DLLS
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
    foreach (file_i ${THIRD_PARTY_DLLS})
        add_custom_command(
                TARGET VulkanRenderer
                POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                ARGS -E copy ${file_i} ${CMAKE_BINARY_DIR}/
        )
    endforeach (file_i)

    ### INSTALL TARGETS ###
    # VkRender Viewer DLL and libraries
    install(TARGETS MultiSense-Viewer CONFIGURATIONS Release RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS MultiSense-Viewer CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS MultiSense-Viewer CONFIGURATIONS Release LIBRARY DESTINATION ${INSTALL_DIRECTORY})

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
            set_property(TARGET MultiSense-Viewer PROPERTY CXX_STANDARD 20)
        endif ()
    endif ()

    # install(TARGETS VkRender-viewer CONFIGURATIONS Release RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})

    file(WRITE ${INSTALL_DIRECTORY}/winreg.ini "")
endif ()


add_custom_command(TARGET MultiSense-Viewer
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/internal/AutoConnect/AutoConnect ${CMAKE_CURRENT_BINARY_DIR}/Assets/Generated/
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Assets/Tools/Ubuntu/StartAutoConnect.sh ${CMAKE_CURRENT_BINARY_DIR}/Assets/Generated/
        )