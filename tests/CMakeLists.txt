# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# unit tests

# tensor
ecbuild_add_test(TARGET        test_mltensor
                 INCLUDES      ${eckit_INCLUDE_DIRS}
                 SOURCES       test_mltensor.cc
                 LIBS          infero eckit
)


# ml engine
ecbuild_add_test(TARGET        test_mlengine
                 INCLUDES      ${eckit_INCLUDE_DIRS}
                 SOURCES       test_mlengine.cc
                 LIBS          infero eckit
)

# regression tests
add_subdirectory(regressions)
