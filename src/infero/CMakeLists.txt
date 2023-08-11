# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

add_subdirectory(api)
add_subdirectory(clustering)
add_subdirectory(models)

# infero.x executable
ecbuild_add_executable( TARGET infero.x
    SOURCES   infero.cc
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS} ${MPI_INCLUDE_DIRS}
    LIBS
        infero
        cnpy
        eckit
        eckit_option
        eckit_mpi
)
