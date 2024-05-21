cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -g")
#
set(CMAKE_C_COMPILER "clang")

project(
        MultiSense-Viewer
        DESCRIPTION "A MultiSense Viewer Application"
        LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set application version. Remember that it should always match the pushed tag in the form of v1.*-*
set(VERSION_MAJOR "1")
set(VERSION_MINOR "1")
set(VERSION_PATCH "4")
set(ARCHITECTURE "amd64")

# Set install directory. Used to generate installers for both Windows and Ubuntu using their native approaches
set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/)

option(BUILD_DOC "Build documentation" ON)
# Set a default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE
            Release
            CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui, ccmake
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"  "MinSizeRel" "RelWithDebInfo")
endif ()

# Find Vulkan
find_package(Vulkan REQUIRED)

include(cmake/Common.cmake)
include(cmake/CompilerWarnings.cmake)

find_package(Doxygen)
if(Doxygen_FOUND)
    add_subdirectory(docs)
else()
    message(STATUS "Doxygen not found, not building docs")
endif()


include_directories(include)
include_directories(Assets)
include_directories(SYSTEM external/ffmpeg/include)
include_directories(SYSTEM ${Vulkan_INCLUDE_DIRS})
ExportScriptIncludes()
GenerateVersionFile()

set(LOGGER_SRC
        include/Viewer/Tools/Logger.cpp
)

set(ENGINE_SRC
        include/Viewer/Core/Buffer.cpp
        include/Viewer/Core/Texture.cpp
        include/Viewer/Core/VulkanDevice.cpp
        include/Viewer/Core/VulkanRenderer.cpp
        include/Viewer/Core/VulkanSwapchain.cpp
        include/Viewer/Core/UUID.cpp
        include/Viewer/CRLCamera/CRLPhysicalCamera.cpp
        include/Viewer/ImGui/GuiManager.cpp

        include/Viewer/ModelLoaders/CRLCameraModels.cpp

        include/Viewer/Renderer/Renderer.cpp
        include/Viewer/Renderer/Entity.cpp
        include/Viewer/Renderer/Components/SkyboxGraphicsPipelineComponent.cpp
        include/Viewer/Renderer/Components/DefaultPBRGraphicsPipelineComponent.cpp
        include/Viewer/Renderer/Components/GLTFModelComponent.cpp
        include/Viewer/Renderer/Components/CustomModels.cpp
        include/Viewer/Renderer/Components/GLTFDefines.h
        include/Viewer/Renderer/Components/SecondaryCameraComponent.cpp
        include/Viewer/Renderer/Components/OBJModelComponent.cpp
        include/Viewer/Renderer/Components/RenderComponents/DefaultGraphicsPipelineComponent2.cpp
        include/Viewer/Renderer/Components/CameraGraphicsPipelineComponent.h

        include/Viewer/Tools/StbInclude.cpp
        include/Viewer/CRLCamera/CameraConnection.cpp
        include/Viewer/ImGui/Widgets.cpp
        include/Viewer/ImGui/Layers/LayerFactory.cpp
        include/Viewer/Renderer/UsageMonitor.cpp
        include/Viewer/Core/RendererConfig.cpp
        include/Viewer/Core/ServerConnection.cpp
        include/Viewer/Core/CommandBuffer.cpp
        include/Viewer/ModelLoaders/ImageView.cpp
        include/Viewer/Renderer/ApplicationUserSetting.h
)

set(SCRIPTS_LIB_SRC
        include/Viewer/Scripts/Private/ScriptUtils.h

        include/Viewer/Scripts/Renderer3D/DefaultScene.cpp
        include/Viewer/Scripts/Renderer3D/MultiSense.cpp
        include/Viewer/Scripts/Renderer3D/Example/Example3D.cpp
        include/Viewer/Scripts/Renderer3D/ImageViewer.cpp
        include/Viewer/Scripts/Renderer3D/DataCapture.cpp

        include/Viewer/SYCL/SYCLRayTracer.cpp ## TODO MOVE out of here and into separate library
        include/Viewer/SYCL/GaussianRenderer.cpp ## TODO MOVE out of here and into separate library
)


set(LIBRARIES "Vulkan::Vulkan;glfw;SimpleIni;imgui;tiff;MultiSense;LibAutoConnect;ktx;nlohmann_json::nlohmann_json;httplib::httplib;rosbag_cpp_writer;tinyobjloader;GPUOpen::VulkanMemoryAllocator;tinyply;")

# Copy Assets folders to output directory
file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/Assets/*")
file(COPY ${ASSETS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Assets)

if (UNIX) ## Linux
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message("   INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
        set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}"
                CACHE PATH "default install path" FORCE)
    endif ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

    # GTK for native file dialog
    # Locate pkg-config
    find_package(PkgConfig REQUIRED)
    # Check for GTK+ using pkg-config
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
    # Add GTK+ include directories
    include_directories(${GTK3_INCLUDE_DIRS})
    # Add GTK+ linking directories
    link_directories(${GTK3_LIBRARY_DIRS})
    # Add GTK+ compiler flags
    add_definitions(${GTK3_CFLAGS_OTHER})

    add_library(CustomLoggerInterface INTERFACE)
    target_link_libraries(CustomLoggerInterface INTERFACE fmt::fmt)

    add_library(CustomLogger ${LOGGER_SRC})
    target_link_libraries(CustomLogger PRIVATE CustomLoggerInterface)

    add_library(VulkanRenderer STATIC ${ENGINE_SRC})
    target_link_libraries(VulkanRenderer ${LIBRARIES}  ${GTK3_LIBRARIES} CustomLogger )
    add_dependencies(VulkanRenderer rosbag_cpp_writer)

    add_library(ScriptLibrary STATIC ${SCRIPTS_LIB_SRC})
    target_link_libraries(ScriptLibrary VulkanRenderer -lavutil -lswscale rosbag_cpp_writer )

    add_executable(MultiSense-Viewer include/main.cpp)
    target_link_libraries(MultiSense-Viewer VulkanRenderer ScriptLibrary )
    add_dependencies(MultiSense-Viewer AutoConnect rosbag_cpp_writer)

    set_project_warnings(ScriptLibrary)
    set_project_warnings(VulkanRenderer)
    set_project_warnings(MultiSense-Viewer)


    install(TARGETS MultiSense-Viewer RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/AutoConnect CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/AutoConnectLauncher.sh CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})

    add_custom_command(TARGET MultiSense-Viewer
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/internal/AutoConnect/AutoConnect ${CMAKE_BINARY_DIR}/
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/Assets/Tools/Ubuntu/AutoConnectLauncher.sh ${CMAKE_BINARY_DIR}/
    )
endif ()

if (WIN32)
    # Set install directory if not specified
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message("   INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
        set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}"
                CACHE PATH "default install path" FORCE)
    endif ()

    if (MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
        add_definitions(-DNOMINMAX)
    endif ()

    # Include directory for WinPcap
    include_directories(internal/AutoConnect/include/WinPcap)

    add_library(avutil STATIC IMPORTED GLOBAL)
    set_target_properties(avutil PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES  "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/avutil.lib")

    add_library(swscale STATIC IMPORTED GLOBAL)
    set_target_properties(swscale PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/lib/swscale.lib")

    add_library(libsll STATIC IMPORTED GLOBAL)
    set_target_properties(libsll PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/lib/libssl.lib")

    add_library(libcrypto STATIC IMPORTED GLOBAL)
    set_target_properties(libcrypto PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/lib/libcrypto.lib")

    add_library(CustomLoggerInterface INTERFACE)
    target_link_libraries(CustomLoggerInterface INTERFACE fmt::fmt)

    add_library(CustomLogger ${LOGGER_SRC})
    target_link_libraries(CustomLogger PRIVATE CustomLoggerInterface )

    add_library(VulkanRenderer STATIC ${ENGINE_SRC})
    target_link_libraries(VulkanRenderer ${LIBRARIES} CustomLogger)

    add_library(ScriptLibrary STATIC ${SCRIPTS_LIB_SRC})
    target_link_libraries(ScriptLibrary VulkanRenderer avutil swscale libsll libcrypto rosbag_cpp_writer)

    add_executable(MultiSense-Viewer WIN32 include/main.cpp)
    target_link_libraries(MultiSense-Viewer VulkanRenderer ScriptLibrary CustomLoggerInterface userenv.lib)
    add_dependencies(MultiSense-Viewer AutoConnect)

    set_project_warnings(ScriptLibrary)
    set_project_warnings(VulkanRenderer)
    set_project_warnings(MultiSense-Viewer)

    #AutoConnect
    set_target_properties(MultiSense-Viewer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(MultiSense-Viewer PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE  ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG  ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${CMAKE_BINARY_DIR})
    set_target_properties(AutoConnect PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG  ${CMAKE_BINARY_DIR})
    set_target_properties(MultiSense PROPERTIES RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR})
    set_target_properties(MultiSense PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${CMAKE_BINARY_DIR})
    set_target_properties(MultiSense PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG  ${CMAKE_BINARY_DIR})

    # create a list of third part DLLs to copy
    set(THIRD_PARTY_DLLS
            #FFMPEG
            ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/swscale-6.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/external/ffmpeg/bin/avutil-57.dll
            #OpenSSL
            ${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/bin/libcrypto-1_1-x64.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/bin/libssl-1_1-x64.dll
            # WinPcap DLLs
            ${CMAKE_CURRENT_SOURCE_DIR}/internal/AutoConnect/WinPcap/Packet.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/internal/AutoConnect/WinPcap/wpcap.dll
    )
    # do the copying
    foreach (file_i ${THIRD_PARTY_DLLS})
        add_custom_command(
                TARGET VulkanRenderer
                POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                ARGS -E copy ${file_i} ${CMAKE_BINARY_DIR}/)
    endforeach (file_i)

    ### INSTALL TARGETS ###
    # VkRender Viewer DLL and libraries
    install(TARGETS MultiSense-Viewer CONFIGURATIONS Release RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS MultiSense-Viewer CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS MultiSense-Viewer CONFIGURATIONS Release LIBRARY DESTINATION ${INSTALL_DIRECTORY})
    # LibMultiSense and WinPCap dll
    install(FILES
            ${CMAKE_BINARY_DIR}/MultiSense.dll
            ${CMAKE_BINARY_DIR}/Packet.dll
            ${CMAKE_BINARY_DIR}/wpcap.dll
            ${CMAKE_BINARY_DIR}/swscale-6.dll
            ${CMAKE_BINARY_DIR}/avutil-57.dll
            ${CMAKE_BINARY_DIR}/libcrypto-1_1-x64.dll
            ${CMAKE_BINARY_DIR}/libssl-1_1-x64.dll
            CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})

    # AutoConnect
    install(PROGRAMS ${CMAKE_BINARY_DIR}/AutoConnect.exe CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})
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
endif ()

# Compile definitions
if (CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_definitions(VulkanRenderer PRIVATE MULTISENSE_VIEWER_PRODUCTION)
    target_compile_definitions(ScriptLibrary PRIVATE MULTISENSE_VIEWER_PRODUCTION)
    target_compile_definitions(MultiSense-Viewer PRIVATE MULTISENSE_VIEWER_PRODUCTION)
else()
    target_compile_definitions(VulkanRenderer PRIVATE MULTISENSE_VIEWER_DEBUG)
    target_compile_definitions(ScriptLibrary PRIVATE MULTISENSE_VIEWER_DEBUG)
    target_compile_definitions(MultiSense-Viewer PRIVATE MULTISENSE_VIEWER_DEBUG)
endif()