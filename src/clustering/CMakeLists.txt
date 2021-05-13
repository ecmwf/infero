
list(APPEND clustering_srcs
    ${PROJECT_SOURCE_DIR}/src/clustering/Clustering.h
    ${PROJECT_SOURCE_DIR}/src/clustering/Clustering.cc
    ${PROJECT_SOURCE_DIR}/src/clustering/ClusteringDBscan.h
    ${PROJECT_SOURCE_DIR}/src/clustering/ClusteringDBscan.cc

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
