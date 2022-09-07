message("Linking MultiSense lib")

add_library(MultiSenseWindows STATIC IMPORTED GLOBAL)
set_property(TARGET MultiSenseWindows APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(MultiSenseWindows PROPERTIES
IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
IMPORTED_LOCATION_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/MultiSense.lib")


message("${CMAKE_CURRENT_SOURCE_DIR}/MultiSense.lib")