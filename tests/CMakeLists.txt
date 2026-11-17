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

# TensorFlow configuration serialization can be tested without the backend.
ecbuild_add_test(TARGET        infero_test_tfc_session_config
                 INCLUDES      ${eckit_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src
                 SOURCES       test_tfc_session_config.cc
                 LIBS          eckit
)

if(HAVE_TESTS AND HAVE_TF_C)
  target_compile_definitions(infero_test_tfc_session_config PRIVATE HAVE_TF_C)
  target_include_directories(infero_test_tfc_session_config PRIVATE ${TENSORFLOWC_INCLUDE_DIRS})
  target_link_libraries(infero_test_tfc_session_config ${TENSORFLOWC_LIBRARIES})
endif()

# regression tests
add_subdirectory(regressions)

# Copy test data to build directory (if tests are enabled)
if(HAVE_TESTS)
  add_custom_command(TARGET infero_test_inference_model POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_directory
                     ${CMAKE_CURRENT_SOURCE_DIR}/data
                     ${CMAKE_BINARY_DIR}/tests/data
                     )
endif()

# c-api
add_subdirectory(c_api)


# f-api
if(infero_HAVE_FORTRAN_APIS)
    add_subdirectory(f_api)
endif()
