cmake_minimum_required(VERSION 3.25)

option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" TRUE)
option(ENABLE_SYCL "Compile with SYCL support. Provided that llvm-sycl clang++ is in path" FALSE)

if (ENABLE_SYCL)
    #set(CMAKE_CXX_COMPILER "icpx")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -g -O0")  # Enable SSE4.1 and SSSE3 support. _mm_min_epi32 and _mm_shuffle_epi8 require specific SSE
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -g")
    set(CMAKE_CXX_COMPILER "clang++")

    add_compile_definitions(SYCL_ENABLED)
    message(STATUS "SYCL support enabled. Compiler: ${CMAKE_CXX_COMPILER} Flags: ${CMAKE_CXX_FLAGS}")
else()
    message(STATUS "SYCL support not enabled.")
endif()


project(
        MultiSense-Viewer
        DESCRIPTION "A MultiSense Viewer Application"
        LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set application version
set(VERSION_MAJOR "1")
set(VERSION_MINOR "2")
set(VERSION_PATCH "0")
set(ARCHITECTURE "amd64")
set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}_${ARCHITECTURE}/)

option(BUILD_DOC "Build documentation" ON)
option(PROPRIETARY_GIGEVISION_MODULE "Include the proprietary GigE-Vision module" OFF)
set(GIGEVISION_MODULE_PATH "" CACHE PATH "Path to the proprietary GigE-Vision module")

# Set default build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE
            Release
            CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Find Vulkan
find_package(Vulkan REQUIRED)

include(cmake/Common.cmake)
include(cmake/CompilerWarnings.cmake)

find_package(Doxygen)
if (Doxygen_FOUND)
    add_subdirectory(docs)
else ()
    message(STATUS "Doxygen not found, not building docs")
endif ()

include_directories(src)
include_directories(Assets)
include_directories(SYSTEM external/ffmpeg/include)
include_directories(SYSTEM ${Vulkan_INCLUDE_DIRS})
GenerateVersionFile()

set(SOURCE_FILES
        src/main.cpp
        src/Viewer/Core/Buffer.cpp
        src/Viewer/Core/Texture.cpp
        src/Viewer/Core/VulkanDevice.cpp
        src/Viewer/Core/VulkanRenderer.cpp
        src/Viewer/Core/VulkanSwapchain.cpp
        src/Viewer/Core/UUID.cpp
        src/Viewer/Core/RendererConfig.cpp
        src/Viewer/Core/ServerConnection.cpp
        src/Viewer/Core/CommandBuffer.cpp

        src/Viewer/Renderer/Renderer.cpp
        src/Viewer/Renderer/Entity.cpp
        src/Viewer/Renderer/Components/SkyboxGraphicsPipelineComponent.cpp
        src/Viewer/Renderer/Components/DefaultPBRGraphicsPipelineComponent.cpp
        src/Viewer/Renderer/Components/GLTFModelComponent.cpp
        src/Viewer/Renderer/Components/CustomModels.cpp
        src/Viewer/Renderer/Components/GLTFDefines.h
        src/Viewer/Renderer/Components/SecondaryCameraComponent.cpp
        src/Viewer/Renderer/Components/OBJModelComponent.cpp
        src/Viewer/Renderer/Components/RenderComponents/DefaultGraphicsPipelineComponent2.cpp
        src/Viewer/Renderer/Components/CameraGraphicsPipelineComponent.h
        src/Viewer/Renderer/UsageMonitor.cpp
        src/Viewer/Renderer/ApplicationUserSetting.h

        src/Viewer/ImGui/Widgets.cpp
        src/Viewer/ImGui/GuiManager.cpp
        src/Viewer/ImGui/AdditionalWindows/DebugWindow.h
        src/Viewer/ImGui/Layers/MultiSenseLayerSupport/Sidebar.h
        src/Viewer/ImGui/Layers/MultiSenseLayerSupport/WelcomeScreen.h
        src/Viewer/ImGui/Layers/MultiSenseLayerSupport/ConfigurationPage.h
        src/Viewer/ImGui/Layers/LayerSupport/LayerFactory.cpp

        src/Viewer/Tools/StbInclude.cpp
        src/Viewer/Tools/Logger.cpp
        src/Viewer/Tools/ThreadPool.h

        src/Viewer/Scripts/ScriptSupport/ScriptUtils.h

        src/Viewer/Modules/LibMultiSense/MultiSenseRendererBridge.cpp
        src/Viewer/Modules/LibMultiSense/CommonHeader.h
        src/Viewer/Modules/LibMultiSense/LibMultiSenseConnector.cpp
        src/Viewer/Modules/LibMultiSense/MultiSenseTaskManager.cpp
)

set (SCRIPT_FILES
        src/Viewer/Scripts/Renderer3D/DefaultScene.cpp
        src/Viewer/Scripts/Renderer3D/Example/Example3D.cpp
        src/Viewer/Scripts/Renderer3D/MultiSense.cpp
)
# Define the libraries variable
set(LIBRARIES
        Vulkan::Vulkan
        glfw
        imgui
        tiff
        ktx
        nlohmann_json::nlohmann_json
        httplib::httplib
        tinyobjloader
        GPUOpen::VulkanMemoryAllocator
        tinyply
        MultiSense
)

if (ENABLE_SYCL)
    list(APPEND SOURCE_FILES src/Viewer/SYCL/GaussianRenderer.cpp)
    list(APPEND SCRIPT_FILES src/Viewer/Scripts/Renderer3D/ImageViewer.cpp)

    message("[INFO] SYCL: Adding Gaussian Renderer to source")

    target_compile_options(ktx PRIVATE -msse4.1 -mssse3)
    list(APPEND LIBRARIES sycl)

endif()


ExportScriptIncludes(${SCRIPT_FILES})

# Copy Assets folders to output directory
file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/Assets/*")
file(COPY ${ASSETS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Assets)

if (UNIX) # Linux
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message("INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
        set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}"
                CACHE PATH "default install path" FORCE)
    endif ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

    # GTK for native file dialog
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
    include_directories(${GTK3_INCLUDE_DIRS})
    link_directories(${GTK3_LIBRARY_DIRS})
    add_definitions(${GTK3_CFLAGS_OTHER})

    add_executable(MultiSense-Viewer ${SOURCE_FILES})
    target_link_libraries(MultiSense-Viewer ${LIBRARIES} ${GTK3_LIBRARIES} fmt::fmt)

    set_project_warnings(MultiSense-Viewer)

    if (PROPRIETARY_GIGEVISION_MODULE)
        target_link_libraries(MultiSense-Viewer ${PROPRIETARY_MODULE_LIB})
    endif ()

    install(TARGETS MultiSense-Viewer RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})
endif ()

if (WIN32)
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message("INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
        set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}"
                CACHE PATH "default install path" FORCE)
    endif ()

    if (MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
        add_definitions(-DNOMINMAX)
    endif ()

    add_library(libsll STATIC IMPORTED GLOBAL)
    set_target_properties(libsll PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/lib/libssl.lib")

    add_library(libcrypto STATIC IMPORTED GLOBAL)
    set_target_properties(libcrypto PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES "C"
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/lib/libcrypto.lib")

    add_executable(MultiSense-Viewer WIN32 ${SOURCE_FILES})
    target_link_libraries(MultiSense-Viewer ${LIBRARIES} libsll libcrypto fmt::fmt userenv.lib)

    set_project_warnings(MultiSense-Viewer)

    set_target_properties(MultiSense-Viewer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(MultiSense-Viewer PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})

    set(THIRD_PARTY_DLLS
            ${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/bin/libcrypto-1_1-x64.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/external/openssl_1.1.1/bin/libssl-1_1-x64.dll
    )

    foreach (file_i ${THIRD_PARTY_DLLS})
        add_custom_command(
                TARGET MultiSense-Viewer
                POST_BUILD
                COMMAND ${CMAKE_COMMAND}
                ARGS -E copy ${file_i} ${CMAKE_BINARY_DIR}/)
    endforeach ()

    install(TARGETS MultiSense-Viewer CONFIGURATIONS Release RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS MultiSense-Viewer CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS MultiSense-Viewer CONFIGURATIONS Release LIBRARY DESTINATION ${INSTALL_DIRECTORY})

    install(FILES
            ${CMAKE_BINARY_DIR}/Packet.dll
            ${CMAKE_BINARY_DIR}/wpcap.dll
            ${CMAKE_BINARY_DIR}/swscale-6.dll
            ${CMAKE_BINARY_DIR}/avutil-57.dll
            ${CMAKE_BINARY_DIR}/libcrypto-1_1-x64.dll
            ${CMAKE_BINARY_DIR}/libssl-1_1-x64.dll
            CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})

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
    target_compile_definitions(MultiSense-Viewer PRIVATE VKRENDER_MULTISENSE_VIEWER_PRODUCTION)
    if (PROPRIETARY_GIGEVISION_MODULE)
        target_compile_definitions(MultiSense-Viewer PRIVATE VKRENDER_GIGEVISION_ENABLED)
    endif()
    else ()
    target_compile_definitions(MultiSense-Viewer PRIVATE VKRENDER_MULTISENSE_VIEWER_DEBUG)
endif ()
