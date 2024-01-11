# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

list(APPEND clustering_srcs
    Clustering.h
    Clustering.cc
    ClusteringDBscan.h
    ClusteringDBscan.cc
)


# Clustering library
ecbuild_add_library(

    TARGET cluster

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
        cnpy
        dbscan
)

ecbuild_add_executable(

    TARGET cluster.x

    SOURCES   cluster.cc

    CONDITION HAVE_TOOLS

    INCLUDES  ${eckit_INCLUDE_DIRS}

    LIBS
        cluster
        infero
        eckit
        eckit_option
        cnpy
)


