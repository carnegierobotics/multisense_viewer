

# ExportScriptIncludes Generates ScriptHeader.h and Scripts.txt for automatic import of the script functionality in the viewer.
function(ExportScriptIncludes)
    string(TIMESTAMP Today)
    file(GLOB_RECURSE  SCRIPT_HEADERS RELATIVE "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/Viewer/Scripts/Objects/*.h")
    file(WRITE ${CMAKE_SOURCE_DIR}/Assets/Generated/ScriptHeader.h "// Generated from Cmake ${Today} \n")
    file(WRITE ${CMAKE_SOURCE_DIR}/Assets/Generated/Scripts.txt "# Generated from Cmake ${Today} \n")
    message("INF " ${SCRIPT_HEADERS})
    foreach (Src ${SCRIPT_HEADERS})
        file(APPEND ${CMAKE_SOURCE_DIR}/Assets/Generated/ScriptHeader.h "\#include \"${Src}\"\n")
    endforeach(Src ${SCRIPT_HEADERS})

    foreach (Src ${SCRIPT_HEADERS})
        string(REGEX MATCH "[^\\/]+$" var ${Src})
        string(REGEX MATCH "^[^.]+" res ${var})
        file(APPEND ${CMAKE_SOURCE_DIR}/Assets/Generated/Scripts.txt ${res} \n)
    endforeach(Src ${SCRIPT_HEADERS})

endfunction()
