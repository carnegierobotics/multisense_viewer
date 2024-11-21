cmake_minimum_required(VERSION 3.20)

# Project definition
project(MultiSense-Viewer
        VERSION 1.2.0
        DESCRIPTION "A MultiSense Viewer Application"
        LANGUAGES C CXX
)

# Options
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" FALSE)
option(ENABLE_SYCL "Compile with SYCL support. Provided that llvm-sycl clang++ is in path" FALSE)
option(ENABLE_PYTORCH "Compile with PyTorch" FALSE)
option(BUILD_DOC "Build documentation" ON)
option(PROPRIETARY_GIGEVISION_MODULE "Include the proprietary GigE-Vision module" OFF)
option(SYCL_BACKEND "Select SYCL backend (dpcpp or adaptivecpp)" "adaptivecpp")
option(DPCPP_COMPILER_PATH "If Intel's DPCPP compiler is selected, the compiler path must be set")

# Paths for proprietary module
set(GIGEVISION_LIB_PATH "" CACHE PATH "Path to the proprietary GigE-Vision module library")
set(GIGEVISION_HEADERS_PATH "" CACHE PATH "Path to the proprietary GigE-Vision module headers")

# Compiler settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME})

# Enable CUDA language support and set CUDA standard
# Conditional setup for PyTorch and CUDA
if (ENABLE_PYTORCH)

    # Enable CUDA only if PyTorch is enabled
    enable_language(CUDA)  # Enable CUDA compiler
    set(CMAKE_CUDA_STANDARD 20)
    target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

    # Find the Torch package
    find_package(Torch REQUIRED HINTS /home/magnus/libraries/libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124/libtorch)

    list(APPEND CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/libcupti.so)

    # Link Torch libraries and enable C++20 for the target
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES}     ${CUDA_LIBRARIES}
    )
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
endif ()

# SYCL Backend selection
if (ENABLE_SYCL)
    if (SYCL_BACKEND STREQUAL "dpcpp")
        #set(CMAKE_CXX_COMPILER "/home/magnus/sycl_workspace/llvm/build/bin/clang++")
        set(CMAKE_CXX_COMPILER /opt/intel/oneapi/compiler/2025.0/bin/icpx) ##TODO FUll path to compiler must be set as argument instead

        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -g -O0" )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -g -O0 -ltbb -lur_loader ")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64 -ltbb")

        message(STATUS "[VkRenderINFO]: SYCL DPC++ support enabled. Compiler: ${CMAKE_CXX_COMPILER} Flags: ${CMAKE_CXX_FLAGS}")
        #target_link_libraries(${PROJECT_NAME} PRIVATE sycl ur_loader)

    elseif (SYCL_BACKEND STREQUAL "adaptivecpp")
        # Find the AdaptiveCpp package
        find_package(AdaptiveCpp REQUIRED)
        # Check if ACPP_TARGETS is passed via command line or environment
        #set(ACPP_TARGETS "omp")
        # Print a message showing which targets are being used
        message(STATUS "[VkRenderINFO]: Using AdaptiveCpp with targets: ${ACPP_TARGETS} to project name ${PROJECT_NAME}")
        add_sycl_to_target(TARGET ${PROJECT_NAME})

        # Add sycl to target using AdaptiveCpp's helper
        target_link_libraries(${PROJECT_NAME} PRIVATE -ltbb -fopenmp -lfmt)
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_options(${PROJECT_NAME} PRIVATE -g --acpp-targets=omp)
            message(STATUS "[VkRenderINFO]: Compiling using AdaptiveCpp without optimization")
        endif ()
    else ()
        message(FATAL_ERROR "[VkRenderERROR]: Unknown SYCL backend. Choose either 'dpcpp' or 'adaptivecpp'.")
    endif ()
    target_compile_definitions(${PROJECT_NAME} PRIVATE SYCL_ENABLED)
    message(STATUS "[VkRenderINFO] Added definition: SYCL_ENABLED")

else ()
    message(STATUS "[VkRenderINFO]: SYCL support not enabled.")
endif ()

# Set default build type if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "[VkRenderINFO]: Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

# Include common CMake modules
include(cmake/Common.cmake)
include(cmake/CompilerWarnings.cmake)

# Add version file generation
GenerateVersionFile()

# Find Vulkan
find_package(Vulkan REQUIRED)

# Include directories
include_directories(src)
include_directories(Resources/Assets)
include_directories(SYSTEM external/ffmpeg/include)
include_directories(SYSTEM ${Vulkan_INCLUDE_DIRS})
include_directories(SYSTEM src/Viewer/VkRender/ImGui/Custom)

# Find Doxygen for documentation
find_package(Doxygen)
if (Doxygen_FOUND AND BUILD_DOC)
    add_subdirectory(docs)
else ()
    message(STATUS "[VkRenderINFO]: Doxygen not found, not building docs")
endif ()

# Set application version
set(INSTALL_DIRECTORY ${CMAKE_BINARY_DIR}/multisense_${PROJECT_VERSION}_${CMAKE_SYSTEM_PROCESSOR}/)

set(SOURCE_FILES
        src/main.cpp
        src/Viewer/VkRender/Core/Buffer.cpp
        src/Viewer/VkRender/Core/VulkanDevice.cpp
        src/Viewer/VkRender/Core/VulkanRenderer.cpp
        src/Viewer/VkRender/Core/VulkanSwapchain.cpp
        src/Viewer/VkRender/Core/UUID.cpp
        src/Viewer/VkRender/Core/ServerConnection.cpp
        src/Viewer/VkRender/Core/CommandBuffer.cpp
        src/Viewer/VkRender/Core/VulkanRenderPass.cpp
        src/Viewer/VkRender/Core/VulkanResourceManager.cpp
        src/Viewer/VkRender/Core/VulkanGraphicsPipeline.cpp
        src/Viewer/VkRender/Core/VulkanImage.cpp
        src/Viewer/VkRender/Core/VulkanFramebuffer.cpp
        src/Viewer/VkRender/Core/VulkanTexture.cpp
        src/Viewer/VkRender/Core/VulkanTexture.h
        src/Viewer/VkRender/Core/Entity.cpp

        src/Viewer/Application/ApplicationConfig.cpp
        src/Viewer/Application/pch.cpp
        src/Viewer/Application/Application.cpp
        src/Viewer/Application/UsageMonitor.cpp
        src/Viewer/Application/ApplicationUserSettings.h

        src/Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.cpp
        src/Viewer/VkRender/RenderResources/GraphicsPipeline2D.cpp

        src/Viewer/Scenes/Scene.cpp
        src/Viewer/Scenes/SceneSerializer.cpp

        src/Viewer/VkRender/Components/MeshComponent.cpp
        src/Viewer/VkRender/Components/ImageComponent.cpp
        src/Viewer/VkRender/Components/PointCloudComponent.h

        src/Viewer/VkRender/ImGui/Widgets.cpp
        src/Viewer/VkRender/ImGui/GuiManager.cpp
        src/Viewer/VkRender/ImGui/AdditionalWindows/DebugWindow.h
        src/Viewer/VkRender/ImGui/LayerFactory.cpp
        src/Viewer/VkRender/ImGui/GuiResources.cpp

        src/Viewer/VkRender/Editors/Editor.cpp
        src/Viewer/VkRender/Editors/Common/3DViewport/Editor3DViewport.cpp
        src/Viewer/VkRender/Editors/Common/ImageEditor/EditorImage.cpp
        src/Viewer/VkRender/Editors/MainContextLayer.h
        src/Viewer/VkRender/Editors/MultiSenseViewer/Layers/LayoutSettingsLayer.h
        src/Viewer/VkRender/Editors/Video/VideoSource.cpp
        src/Viewer/VkRender/Editors/Video/VideoPlaybackSystem.cpp
        src/Viewer/VkRender/Editors/Common/SceneRenderer.cpp
        src/Viewer/VkRender/Editors/Common/SceneHierarchy/SceneHierarchyLayer.cpp
        src/Viewer/VkRender/Editors/PipelineKey.cpp
        src/Viewer/VkRender/Editors/Common/Properties/PropertiesLayer.cpp
        src/Viewer/VkRender/Editors/PipelineManager.cpp
        src/Viewer/VkRender/Editors/DescriptorSetManager.cpp

        src/Viewer/Tools/StbInclude.cpp
        src/Viewer/Tools/Logger.cpp
        src/Viewer/Tools/ThreadPool.h

        src/Viewer/Modules/MultiSense/MultiSenseRendererBridge.cpp
        src/Viewer/Modules/MultiSense/MultiSenseDevice.cpp
        src/Viewer/Modules/MultiSense/CommonHeader.h
        src/Viewer/Modules/MultiSense/MultiSenseTaskManager.cpp
        src/Viewer/Modules/MultiSense/LibMultiSense/LibMultiSenseConnector.cpp
        src/Viewer/Application/ApplicationUserSettingsSerializer.cpp
        src/Viewer/Application/ApplicationUserSettingsSerializer.h
)

# Libraries
set(LIBRARIES
        Vulkan::Vulkan
        glfw
        imgui
        tiff
        nlohmann_json::nlohmann_json
        httplib::httplib
        tinyobjloader
        GPUOpen::VulkanMemoryAllocator
        tinyply
        MultiSense
        yaml-cpp
        fmt::fmt
)

# Add SYCL-specific files
if (ENABLE_SYCL)
    set(SYCL_SOURCES
            src/Viewer/VkRender/RenderResources/2DGS/SYCLGaussian2D.cpp
            src/Viewer/VkRender/RenderResources/3DGS/SYCLGaussian3D.cpp
            src/Viewer/VkRender/Core/SyclDeviceSelector.h
            src/Viewer/VkRender/Editors/Common/GaussianViewer/EditorGaussianViewer.cpp
    )
    list(APPEND SOURCE_FILES ${SYCL_SOURCES})
    message(STATUS "[VkRenderINFO]: SYCL: Adding Gaussian Renderer to source")

    if (${ENABLE_PYTORCH})
        set (DIFF_RENDER_PYTORCH         src/Viewer/VkRender/RenderResources/DifferentiableRenderer/RenderGaussianFunction.cpp
                src/Viewer/VkRender/RenderResources/DifferentiableRenderer/RenderGaussian.cpp)
        list(APPEND SOURCE_FILES ${DIFF_RENDER_PYTORCH})
        message(STATUS "[VkRenderINFO]: SYCL: Adding Pytorch differential Renderer to source")

    endif()
else()
    message(STATUS "[VkRenderINFO]: Skipped Adding SYCL Sources")

endif ()# Add SYCL-specific files

# Add proprietary GigE-Vision module
if (PROPRIETARY_GIGEVISION_MODULE)
    list(APPEND SOURCE_FILES src/Viewer/Modules/MultiSense/GigE-Vision/GigEVisionConnector.cpp)
    list(APPEND LIBRARIES ${GIGEVISION_LIB})
    target_compile_definitions(${PROJECT_NAME} PRIVATE VKRENDER_GIGEVISION_ENABLED)
endif ()

#Custom command and target to copy assets if they have changed
file(GLOB ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/Resources/Assets/*")
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Resources/Assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/Resources/Assets ${CMAKE_CURRENT_BINARY_DIR}/Resources/Assets
        DEPENDS ${ASSETS}
        COMMENT "Copying assets..."
)
add_custom_target(copy_assets ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Resources/Assets)

# Build configurations for different platforms
if (UNIX)
    # Linux-specific settings
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message(STATUS "INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
        set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}" CACHE PATH "default install path" FORCE)
    endif ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

    # GTK for native file dialog
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
    include_directories(${GTK3_INCLUDE_DIRS})
    link_directories(${GTK3_LIBRARY_DIRS})
    add_definitions(${GTK3_CFLAGS_OTHER})

    # Build executable
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES} ${GTK3_LIBRARIES} fmt::fmt)
    add_dependencies(${PROJECT_NAME} copy_assets)

    # Set warnings
    set_project_warnings(${PROJECT_NAME})

    # Install configuration
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/Resources/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})

elseif (WIN32)
    # Windows-specific settings
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        message(STATUS "INFO: Default Install prefix was chosen so set install path to ${INSTALL_DIRECTORY}")
        set(CMAKE_INSTALL_PREFIX "${INSTALL_DIRECTORY}" CACHE PATH "default install path" FORCE)
    endif ()

    if (MSVC)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
        add_definitions(-DNOMINMAX)
    endif ()

    # Build executable
    target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})
    target_link_libraries(${PROJECT_NAME} ${LIBRARIES} fmt::fmt userenv.lib)
    add_dependencies(${PROJECT_NAME} copy_assets)

    # Set warnings
    set_project_warnings(${PROJECT_NAME})

    # Set runtime output directory
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
    set_target_properties(MultiSense PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    # Install configuration
    install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release RUNTIME DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release LIBRARY DESTINATION ${INSTALL_DIRECTORY})
    install(FILES ${CMAKE_BINARY_DIR}/MultiSense.dll CONFIGURATIONS Release ARCHIVE DESTINATION ${INSTALL_DIRECTORY})
    install(DIRECTORY Resources/Assets CONFIGURATIONS Release DESTINATION ${INSTALL_DIRECTORY})

endif ()

# Compile definitions for different build types
if (CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_definitions(${PROJECT_NAME} PRIVATE VKRENDER_MULTISENSE_VIEWER_PRODUCTION)
else ()
    target_compile_definitions(${PROJECT_NAME} PRIVATE VKRENDER_MULTISENSE_VIEWER_DEBUG)
endif ()
