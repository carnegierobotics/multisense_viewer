cmake_minimum_required(VERSION 3.15)
project(AutoConnect LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

set(MULTISENSE_SRC ../../external/LibMultiSense/source/LibMultiSense)
add_subdirectory(${MULTISENSE_SRC}  [EXCLUDE_FROM_ALL])

add_executable(AutoConnect AutoConnect.cpp)
target_link_libraries(AutoConnect MultiSense)