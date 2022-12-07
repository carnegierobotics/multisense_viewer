 # set input and output files
 set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
 set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

 # request to configure the file
 configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
 message("Doxygen build started")

 doxygen_add_docs(viewer-docs
         ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include
         ALL
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMENT "Generating API documentation with Doxygen")