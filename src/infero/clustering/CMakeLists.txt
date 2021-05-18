
list(APPEND clustering_srcs

    Clustering.h
    Clustering.cc
    ClusteringDBscan.h
    ClusteringDBscan.cc

    ${PROJECT_SOURCE_DIR}/contrib/DBSCAN/dbscan.h
    ${PROJECT_SOURCE_DIR}/contrib/DBSCAN/dbscan.cpp
)


# Clustering library
ecbuild_add_library(

    TARGET MLclustering

    INSTALL_HEADERS LISTED

    HEADER_DESTINATION
        ${INSTALL_INCLUDE_DIR}/clustering

    # GENERATED
    #     infero_version.cc

    SOURCES
        ${clustering_srcs}

    PUBLIC_INCLUDES
       $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
       $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
       $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/contrib>

    PRIVATE_INCLUDES
        "${eckit_INCLUDE_DIRS}"

    PRIVATE_LIBS
        infero
        eckit
        eckit_option
        ${ZLIB_LIBRARIES}
)

ecbuild_add_executable( TARGET cluster.x
    SOURCES   cluster.cc
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS}
    LIBS      MLclustering infero eckit eckit_option )
