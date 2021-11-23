ecbuild_add_library( TARGET inferoapi
    INSTALL_HEADERS LISTED
    HEADER_DESTINATION
        ${INSTALL_INCLUDE_DIR}/infero/api
    SOURCES
        infero-api.cc
    PUBLIC_INCLUDES
       $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
       $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    PRIVATE_INCLUDES
        "${eckit_INCLUDE_DIRS}"
    PRIVATE_LIBS
        eckit
    PUBLIC_LIBS
        infero
)

#ecbuild_add_library( TARGET inferof
#    INSTALL_HEADERS LISTED
#    HEADER_DESTINATION
#        ${INSTALL_INCLUDE_DIR}/infero/api
#    SOURCES
#        inferof.F90
#    PUBLIC_INCLUDES
#       $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
#       $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
#    PRIVATE_INCLUDES
#        "${eckit_INCLUDE_DIRS}"
#    PRIVATE_LIBS
#        inferoapi
#        eckit
#)

add_subdirectory(examples)
