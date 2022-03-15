# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# unit tests


# ml engine
ecbuild_add_test(TARGET        infero_test_inference_model
                 INCLUDES      ${eckit_INCLUDE_DIRS}
                 SOURCES       test_inference_model.cc
                 LIBS          infero eckit
)

# regression tests
add_subdirectory(regressions)

# copy test data to build directory
add_custom_command(TARGET infero_test_inference_model POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/data
                   ${CMAKE_BINARY_DIR}/tests/data
                   )

# c-api
add_subdirectory(c_api)


# f-api
if(HAVE_FCKIT)
    add_subdirectory(f_api)
endif()
