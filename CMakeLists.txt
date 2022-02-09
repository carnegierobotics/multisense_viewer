cmake_minimum_required(VERSION 3.21)
project(MultiSense)

set(CMAKE_CXX_STANDARD 20)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenCV 4.0.0 REQUIRED)


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

