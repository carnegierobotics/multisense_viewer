cmake_minimum_required(VERSION 3.2)
project(MultiSense C CXX)

set(CMAKE_CXX_STANDARD 20)


set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE STRING "" )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} CACHE STRING "" )

if (WIN32)

    if(NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
    endif()

    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")

    if( MSVC )
        SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup" )
    endif()

    # GLFW
    set(GLFW_DIR external/glfw) # Set this to point to an up-to-date GLFW repo
    option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
    option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
    option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
    option(GLFW_INSTALL "Generate installation target" OFF)
    option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
    add_subdirectory(${GLFW_DIR} binary_dir EXCLUDE_FROM_ALL)
    include_directories(${GLFW_DIR}/include)


    # Dear ImGui
    set(IMGUI_DIR external/imgui)
    include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

    # GLM
    set(GLM_DIR external/glm)
    include_directories(${GLM_DIR})

    # Libraries
    find_package(Vulkan REQUIRED)   
    #find_library(VULKAN_LIBRARY
      #NAMES vulkan vulkan-1)
    #set(LIBRARIES "glfw;${VULKAN_LIBRARY}")
    set(LIBRARIES "glfw;Vulkan::Vulkan")

    # Use vulkan headers from glfw:
    include_directories(${GLFW_DIR}/deps)
        
    # add source files
    file(GLOB ENGINE_SRC src/core/* src/tools/* src/imgui/* src/Renderer/* src/scripts/*)

    add_executable(MultiSense WIN32 main.cpp  ${ENGINE_SRC} ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_demo.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp)
    target_link_libraries(MultiSense ${LIBRARIES})
    if( MSVC )
        if(${CMAKE_VERSION} VERSION_LESS "3.6.0") 
            message( "\n\t[ WARNING ]\n\n\tCMake version lower than 3.6.\n\n\t - Please update CMake and rerun; OR\n\t - Manually set 'GLFW-CMake-starter' as StartUp Project in Visual Studio.\n" )
        else()
            set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MultiSense )
        endif()
    endif()

    # Copy dependency folders to output directory
    add_custom_target(copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/scripts ${CMAKE_CURRENT_BINARY_DIR}/scripts)
    add_dependencies(MultiSense copy_assets)

else()

    find_package(Vulkan REQUIRED)
    find_package(glfw3 REQUIRED)

    # DEAR IMGUI LIBRARY
    set(IMGUI_DIR external/imgui)
    include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)
    message(STATUS "Adding GUI library files")
    FILE(GLOB GUI_SRC  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp ${IMGUI_DIR}/imgui.cpp ${IMGUI_DIR}/imgui_draw.cpp ${IMGUI_DIR}/imgui_demo.cpp ${IMGUI_DIR}/imgui_tables.cpp ${IMGUI_DIR}/imgui_widgets.cpp)
    add_library(GUI ${GUI_SRC})


    # Create a list of Libraries in project
    set(LIBRARIES glfw vulkan GUI)
    message(LIBRARIES:  ${LIBRARIES})

    file(GLOB ENGINE_SRC src/core/* src/tools/* src/imgui/* src/Renderer/*)
    add_library(VulkanRenderer ${ENGINE_SRC} src/imgui/VkGUI.h)
    target_link_libraries(VulkanRenderer ${LIBRARIES})
    set_target_properties(VulkanRenderer PROPERTIES LINKER_LANGUAGE CXX)

    add_executable(MultiSense main.cpp)
    target_link_libraries(MultiSense VulkanRenderer)


endif()
