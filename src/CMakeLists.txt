# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# infero-version executable
ecbuild_add_executable( TARGET infero-version
    SOURCES   infero_version.cc
    CONDITION HAVE_TOOLS
    INCLUDES  infero_version.h
)

add_subdirectory(infero)
