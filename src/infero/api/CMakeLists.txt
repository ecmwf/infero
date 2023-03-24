if(HAVE_FINAL)
    add_definitions(-DHAVE_FINAL)
endif()

# Infero C-API
ecbuild_add_library(

    TARGET inferoapi

    INSTALL_HEADERS LISTED

    HEADER_DESTINATION
        ${INSTALL_INCLUDE_DIR}/infero/api

    SOURCES
        infero.h
        infero-api.cc

    INSTALL_HEADERS_LIST
        infero.h

    PUBLIC_INCLUDES
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>

    PRIVATE_INCLUDES
        "${eckit_INCLUDE_DIRS}"
        "${FCKIT_INCLUDE_DIRS}"

    PRIVATE_LIBS
        eckit
        fckit

    PUBLIC_LIBS
        infero
)
install( FILES ${CMAKE_CURRENT_SOURCE_DIR}/infero.h DESTINATION include/infero/api )


# Infero Fortran-API
if(HAVE_FCKIT)

ecbuild_add_library(

    TARGET inferof

    INSTALL_HEADERS LISTED

    HEADER_DESTINATION
        ${INSTALL_INCLUDE_DIR}/infero/api

    SOURCES
        inferof.F90

    PUBLIC_INCLUDES
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:module/infero>
        "${FCKIT_INCLUDE_DIRS}"

    PRIVATE_INCLUDES
        "${eckit_INCLUDE_DIRS}"        

    PRIVATE_LIBS
        inferoapi
        eckit        

    PUBLIC_LIBS
        ${FCKIT_LIBRARIES}
)

install( DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/${CMAKE_CFG_INTDIR}
         DESTINATION module/infero
         COMPONENT modules )

endif()


#add_subdirectory(examples)
