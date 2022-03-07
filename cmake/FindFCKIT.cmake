# (C) Copyright 2011- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

# - Try to find FCKIT
# Once done this will define
#
#  FCKIT_FOUND         - found FCKIT
#  FCKIT_INCLUDE_DIRS  - the FCKIT include directories
#  FCKIT_LIBRARIES     - the FCKIT libraries
#
# The following paths will be searched with priority if set in CMake or env
#
#  FCKIT_PATH          - prefix path of the FCKIT installation
#  FCKIT_ROOT          - Set this variable to the root installation

# Search with priority for FCKIT_PATH if given as CMake or env var
find_path(FCKIT_INCLUDE_DIR fckit/fckit.h
          HINTS $ENV{FCKIT_ROOT} ${FCKIT_ROOT}
          PATHS ${FCKIT_PATH} ENV FCKIT_PATH
          PATH_SUFFIXES include NO_DEFAULT_PATH)

find_path(FCKIT_INCLUDE_DIR fckit.h PATH_SUFFIXES include )

# Search with priority for FCKIT_PATH if given as CMake or env var
find_library(FCKIT_LIB fckit
            HINTS $ENV{FCKIT_ROOT}
            PATHS ${FCKIT_PATH} ENV FCKIT_PATH
            PATH_SUFFIXES lib64 lib NO_DEFAULT_PATH)

find_library(FCKIT_LIB fckit PATH_SUFFIXES lib64 lib )

set( FCKIT_LIBRARIES    ${FCKIT_LIB} )
set( FCKIT_INCLUDE_DIRS ${FCKIT_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

# handle the QUIET and REQUIRED arguments and set FCKIT_FOUND to TRUE
# if all listed variables are TRUE
# Note: capitalisation of the package name must be the same as in the file name
find_package_handle_standard_args(FCKIT DEFAULT_MSG FCKIT_LIB FCKIT_INCLUDE_DIR)

mark_as_advanced(FCKIT_INCLUDE_DIR FCKIT_LIB)
